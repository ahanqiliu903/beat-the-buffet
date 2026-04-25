"""
Sushi Ingredient Price Checker
==============================
Estimate prices for sushi ingredients in NY cities by querying serper.dev's
/shopping endpoint, parsing the returned product listings, and then pricing
out four sushi recipes from the median ingredient unit prices.

Usage:
    python sushi_prices.py -l "Buffalo" -k YOUR_API_KEY
    python sushi_prices.py -l "Albany" -i Nori Rice Salmon -k YOUR_API_KEY
    python sushi_prices.py -l "New York" -k YOUR_API_KEY -n 8
"""

import argparse
import re
import statistics
import sys
from typing import Optional, Tuple, List, Dict, Any

import requests

SERPER_URL = "https://google.serper.dev/shopping"

VALID_CITIES = {
    "New York", "Buffalo", "Rochester", "Yonkers", "Syracuse",
    "Albany", "Cheektowaga", "New Rochelle", "Mount Vernon",
    "Schenectady", "Utica", "Brentwood", "White Plains",
    "Hamburg", "Niagara Falls",
}

DEFAULT_INGREDIENTS = ["Nori", "Rice", "Salmon", "Tuna", "Shrimp", "Tempura"]

DEFAULT_MAX_ENTRIES = 5
PER_PAGE = 20
MAX_PAGES = 3

# ----------------------------------------------------------------------
# Sushi recipe table
# ----------------------------------------------------------------------
# Each ingredient is encoded as (unit_kind, amount_in_that_unit).
# unit_kind is "oz" (priced per ounce) or "count" (priced per sheet/ct).
# Gram amounts are pre-converted to oz so the math per piece is just a
# multiplication against the median $/oz.
GRAMS_PER_OZ = 28.3495
RICE_OZ_PER_PIECE = 20.0 / GRAMS_PER_OZ        # 20 g rice per piece
NIGIRI_FISH_OZ_PER_PIECE = 15.0 / GRAMS_PER_OZ  # 15 g fish per nigiri

# Keys are lowercase ingredient names (matched against user input).
SUSHI_RECIPES: Dict[str, Dict[str, Tuple[str, float]]] = {
    "Salmon nigiri": {
        "rice":   ("oz", RICE_OZ_PER_PIECE),
        "salmon": ("oz", NIGIRI_FISH_OZ_PER_PIECE),
    },
    "Tuna nigiri": {
        "rice": ("oz", RICE_OZ_PER_PIECE),
        "tuna": ("oz", NIGIRI_FISH_OZ_PER_PIECE),
    },
    "Salmon sashimi": {
        "salmon": ("oz", 1.0),
    },
    "Shrimp tempura roll": {
        "rice":    ("oz", RICE_OZ_PER_PIECE),
        "shrimp":  ("oz", 0.3),
        "tempura": ("oz", 0.3),
        "nori":    ("count", 0.1),
    },
}


# ----------------------------------------------------------------------
# Parsing layer
# ----------------------------------------------------------------------

WEIGHT_UNITS = {
    "oz": ("oz", 1.0),
    "ounce": ("oz", 1.0),
    "ounces": ("oz", 1.0),
    "lb": ("oz", 16.0),
    "lbs": ("oz", 16.0),
    "pound": ("oz", 16.0),
    "pounds": ("oz", 16.0),
    "g": ("g", 1.0),
    "gram": ("g", 1.0),
    "grams": ("g", 1.0),
    "kg": ("g", 1000.0),
    "kilogram": ("g", 1000.0),
    "kilograms": ("g", 1000.0),
}

COUNT_UNITS = {"ct", "count", "pack", "pk", "sheets", "sheet",
               "pieces", "piece", "pcs"}

UNIT_REGEX = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*"
    r"(oz|ounces?|lbs?|pounds?|kg|kilograms?|g(?:rams?)?|"
    r"ct|count|pack|pk|sheets?|pieces?|pcs)\b",
    re.IGNORECASE,
)

PRICE_REGEX = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)")


def parse_price(text: str) -> Optional[float]:
    """Return the first USD price found in ``text``, or None."""
    if not text:
        return None
    m = PRICE_REGEX.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_unit(text: str) -> Optional[Tuple[float, str, str, float]]:
    """
    Extract a measurement from a product description.

    Returns (quantity, raw_unit, family, normalized_quantity), where
    ``family`` is one of {"oz", "g", "count"} and ``normalized_quantity``
    is the value expressed in the family's base unit. None when not found.
    """
    if not text:
        return None
    m = UNIT_REGEX.search(text)
    if not m:
        return None
    qty = float(m.group(1))
    raw_unit = m.group(2).lower().rstrip(".")

    if raw_unit in WEIGHT_UNITS:
        family, factor = WEIGHT_UNITS[raw_unit]
        return (qty, raw_unit, family, qty * factor)
    if raw_unit in COUNT_UNITS:
        return (qty, raw_unit, "count", qty)
    return None


def compute_unit_price(price: Optional[float],
                       parsed_unit: Optional[Tuple]
                       ) -> Optional[Tuple[float, str]]:
    """Return (price_per_base_unit, label) for human-readable per-row output."""
    if price is None or parsed_unit is None:
        return None
    _qty, raw_unit, family, normalized = parsed_unit
    if normalized <= 0:
        return None
    if family == "oz":
        return (price / normalized, "per oz")
    if family == "g":
        return (price / normalized, "per g")
    if family == "count":
        return (price / normalized, f"per {raw_unit}")
    return None


def to_per_oz(price: float, parsed_unit: Tuple) -> Optional[float]:
    """
    Convert a (price, parsed_unit) pair into $/oz if the unit is a weight.
    Returns None for count-based units.
    """
    _qty, _raw, family, normalized = parsed_unit
    if normalized <= 0:
        return None
    if family == "oz":
        return price / normalized
    if family == "g":
        return (price / normalized) * GRAMS_PER_OZ
    return None


def to_per_count(price: float, parsed_unit: Tuple) -> Optional[float]:
    """Convert (price, parsed_unit) into $/count-item, else None."""
    _qty, _raw, family, normalized = parsed_unit
    if family != "count" or normalized <= 0:
        return None
    return price / normalized


# ----------------------------------------------------------------------
# Network layer
# ----------------------------------------------------------------------

def search_serper(query: str, location: str, api_key: str,
                  num: int = PER_PAGE, page: int = 1,
                  timeout: int = 15) -> Dict[str, Any]:
    """POST to serper.dev /shopping and return the raw JSON payload."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {
        "q": f"sushi {query}",
        "location": f"{location}, New York, United States",
        "gl": "us",
        "hl": "en",
        "num": num,
        "page": page,
    }
    r = requests.post(SERPER_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def extract_candidates(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pull product entries from a /shopping response."""
    candidates: List[Dict[str, str]] = []
    for item in payload.get("shopping", []):
        candidates.append({
            "title": item.get("title", "") or "",
            "snippet": item.get("price", "") or "",
            "source": item.get("source", "") or item.get("link", "shopping"),
        })
    return candidates


def estimate_ingredient_price(ingredient: str, location: str, api_key: str,
                              max_entries: int = DEFAULT_MAX_ENTRIES,
                              per_page: int = PER_PAGE,
                              max_pages: int = MAX_PAGES,
                              ) -> Dict[str, Any]:
    """
    Search and keep parsing until ``max_entries`` rows have BOTH price and
    unit. Paginates up to ``max_pages`` times if needed.
    """
    qualifying: List[Dict[str, Any]] = []
    skipped_no_price = 0
    skipped_no_unit = 0
    pages_fetched = 0
    candidates_seen = 0

    for page in range(1, max_pages + 1):
        payload = search_serper(ingredient, location, api_key,
                                num=per_page, page=page)
        pages_fetched += 1
        candidates = extract_candidates(payload)
        if not candidates:
            break

        for c in candidates:
            candidates_seen += 1
            blob = f"{c['title']} {c['snippet']}"
            price = parse_price(blob)
            unit = parse_unit(blob)

            if price is None:
                skipped_no_price += 1
                continue
            if unit is None:
                skipped_no_unit += 1
                continue

            qualifying.append({
                "title": c["title"][:80],
                "source": c["source"],
                "price": price,
                "unit": unit,
                "unit_price": compute_unit_price(price, unit),
            })
            if len(qualifying) >= max_entries:
                break

        if len(qualifying) >= max_entries:
            break

    prices = [r["price"] for r in qualifying]
    return {
        "ingredient": ingredient,
        "location": location,
        "samples": qualifying,
        "target_entries": max_entries,
        "pages_fetched": pages_fetched,
        "candidates_seen": candidates_seen,
        "skipped_no_price": skipped_no_price,
        "skipped_no_unit": skipped_no_unit,
        "median_price": statistics.median(prices) if prices else None,
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
    }


def aggregate_unit_prices(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce one ingredient's qualifying samples to median $/oz (across weight
    parses, gram entries auto-converted) and median $/count (across count
    parses). Returns counts of how many samples contributed to each.
    """
    per_oz: List[float] = []
    per_count: List[float] = []
    for s in result["samples"]:
        oz = to_per_oz(s["price"], s["unit"])
        if oz is not None:
            per_oz.append(oz)
            continue
        ct = to_per_count(s["price"], s["unit"])
        if ct is not None:
            per_count.append(ct)
    return {
        "per_oz": statistics.median(per_oz) if per_oz else None,
        "per_count": statistics.median(per_count) if per_count else None,
        "n_oz": len(per_oz),
        "n_count": len(per_count),
    }


def price_sushi(all_results: Dict[str, Dict[str, Any]]
                ) -> Tuple[Dict[str, Dict[str, Any]],
                           Dict[str, Dict[str, Any]],
                           List[Tuple[str, str]]]:
    """
    Build per-piece prices for any sushi whose ingredients are all available.

    Returns (makeable, unit_prices, skipped):
      - makeable[name] = {"price_per_piece": float, "breakdown": [...]}
      - unit_prices[ingredient] = aggregate_unit_prices(...) result
      - skipped = [(sushi_name, reason), ...] for sushi we couldn't price
    """
    # Match user input by lowercase name; ingredient may be missing entirely
    # (not in --ingredients) or present-but-unusable (no qualifying samples
    # in the family the recipe needs).
    unit_prices = {
        name.lower(): aggregate_unit_prices(res)
        for name, res in all_results.items()
    }

    makeable: Dict[str, Dict[str, Any]] = {}
    skipped: List[Tuple[str, str]] = []

    for sushi_name, recipe in SUSHI_RECIPES.items():
        breakdown = []
        total = 0.0
        reason = None

        for ing, (unit_kind, amount) in recipe.items():
            ip = unit_prices.get(ing)
            if ip is None:
                reason = f"{ing} not searched"
                break

            if unit_kind == "oz":
                rate = ip["per_oz"]
                if rate is None:
                    reason = f"{ing} has no $/oz parses (got {ip['n_count']} count-only samples)"
                    break
                cost = amount * rate
                breakdown.append((ing, amount, "oz", rate, cost))
                total += cost
            elif unit_kind == "count":
                rate = ip["per_count"]
                if rate is None:
                    reason = f"{ing} has no $/sheet parses (got {ip['n_oz']} weight-only samples)"
                    break
                cost = amount * rate
                breakdown.append((ing, amount, "sheet", rate, cost))
                total += cost
            else:
                reason = f"unknown unit kind {unit_kind!r}"
                break

        if reason:
            skipped.append((sushi_name, reason))
        else:
            makeable[sushi_name] = {
                "price_per_piece": total,
                "breakdown": breakdown,
            }

    return makeable, unit_prices, skipped


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------

def print_summary(result: Dict[str, Any]) -> None:
    print(f"\n=== {result['ingredient']} in {result['location']}, NY ===")

    n_got = len(result["samples"])
    n_target = result["target_entries"]
    if n_got < n_target:
        print(f"  ! Only {n_got}/{n_target} qualifying entries found "
              f"(after {result['pages_fetched']} page(s), "
              f"{result['candidates_seen']} candidates seen; "
              f"{result['skipped_no_price']} missing price, "
              f"{result['skipped_no_unit']} missing unit).")

    if result["median_price"] is None:
        print("  No prices parsed from results.")
        return

    print(f"  Median: ${result['median_price']:.2f}  "
          f"(min ${result['min_price']:.2f}, "
          f"max ${result['max_price']:.2f})  "
          f"across {n_got} sample(s)")

    for i, row in enumerate(result["samples"], 1):
        price_str = f"${row['price']:.2f}"
        if row["unit_price"]:
            up_val, up_label = row["unit_price"]
            unit_str = f"~${up_val:.3f} {up_label}"
        else:
            qty, raw_unit, *_ = row["unit"]
            unit_str = f"{qty:g} {raw_unit}"
        print(f"  {i}. {price_str:>8}  {unit_str:<24}  {row['title']}")


def print_sushi_report(location: str,
                       makeable: Dict[str, Dict[str, Any]],
                       unit_prices: Dict[str, Dict[str, Any]],
                       skipped: List[Tuple[str, str]]) -> None:
    print(f"\n=== Sushi pricing in {location}, NY ===")

    if unit_prices:
        print("Median ingredient unit prices:")
        for ing, ip in sorted(unit_prices.items()):
            parts = []
            if ip["per_oz"] is not None:
                parts.append(f"${ip['per_oz']:.3f}/oz (n={ip['n_oz']})")
            if ip["per_count"] is not None:
                parts.append(f"${ip['per_count']:.3f}/sheet (n={ip['n_count']})")
            if not parts:
                parts.append("no usable parses")
            print(f"  {ing:<8} {' | '.join(parts)}")

    if makeable:
        print("\nHypothetical price per piece:")
        for name, info in makeable.items():
            print(f"  {name:<22} ${info['price_per_piece']:.3f}")
            for ing, amount, unit_label, rate, cost in info["breakdown"]:
                print(f"      {ing:<8} {amount:.3f} {unit_label} "
                      f"@ ${rate:.3f}  =  ${cost:.3f}")
    else:
        print("\nNo sushi could be priced from the parsed ingredients.")

    if skipped:
        print("\nSkipped sushi:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate sushi ingredient prices in NY cities via serper.dev",
    )
    p.add_argument("-l", "--location", required=True,
                   help="NY city, e.g. 'Buffalo' or 'New York'.")
    p.add_argument("-i", "--ingredients", nargs="+", default=DEFAULT_INGREDIENTS,
                   help=f"Ingredients to price. Default: {DEFAULT_INGREDIENTS}")
    p.add_argument("-k", "--api-key", required=True, help="serper.dev API key.")
    p.add_argument("-n", "--max-entries", type=int, default=DEFAULT_MAX_ENTRIES,
                   help=f"Qualifying rows wanted per ingredient "
                        f"(default: {DEFAULT_MAX_ENTRIES}).")
    p.add_argument("--per-page", type=int, default=PER_PAGE,
                   help=f"Results per API call (default: {PER_PAGE}).")
    p.add_argument("--max-pages", type=int, default=MAX_PAGES,
                   help=f"Max pages to scan per ingredient (default: {MAX_PAGES}).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.location not in VALID_CITIES:
        print(f"Error: '{args.location}' is not in the known NY city list.",
              file=sys.stderr)
        return 1

    all_results: Dict[str, Dict[str, Any]] = {}
    for ingredient in args.ingredients:
        try:
            result = estimate_ingredient_price(
                ingredient, args.location, args.api_key,
                max_entries=args.max_entries,
                per_page=args.per_page,
                max_pages=args.max_pages,
            )
            print_summary(result)
            if result["samples"]:
                all_results[ingredient] = result
        except requests.HTTPError as e:
            print(f"[{ingredient}] HTTP error: {e}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"[{ingredient}] Network error: {e}", file=sys.stderr)

    makeable, unit_prices, skipped = price_sushi(all_results)
    print_sushi_report(args.location, makeable, unit_prices, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())