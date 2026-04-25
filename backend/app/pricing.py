import os

from scripts.sushi_prices import (
    aggregate_unit_prices,
    estimate_ingredient_price,
    price_sushi,
)

LOCATION = "New York"

# Map canonical ingredient name (matches recipe keys) -> serper search query.
# Generic single words ("Tuna") return too much canned/processed product, so we
# pin queries to sushi-grade / raw retail terms.
INGREDIENT_QUERIES: dict[str, str] = {
    "Nori": "nori seaweed sheets",
    "Rice": "sushi rice",
    "Salmon": "sushi grade salmon",
    "Tuna": "sushi grade ahi tuna",
    "Shrimp": "raw shrimp",
    "Tempura": "tempura batter mix",
}

_cache: dict | None = None
_skipped: list[tuple[str, str]] | None = None


def load_prices() -> bool:
    global _cache, _skipped
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        print("[pricing] SERPER_API_KEY not set; skipping price prefetch")
        return False

    print(f"[pricing] fetching ingredient prices in {LOCATION}, NY...")
    all_results: dict = {}
    for canonical, query in INGREDIENT_QUERIES.items():
        try:
            result = estimate_ingredient_price(query, LOCATION, api_key)
            if result["samples"]:
                all_results[canonical] = result
                up = aggregate_unit_prices(result)
                parts = []
                if up["per_oz"] is not None:
                    parts.append(f"${up['per_oz']:.3f}/oz n={up['n_oz']}")
                if up["per_count"] is not None:
                    parts.append(f"${up['per_count']:.3f}/ct n={up['n_count']}")
                summary = " | ".join(parts) if parts else "no usable parses"
                print(f"[pricing] {canonical:<8} ({query}): {summary}")
            else:
                print(f"[pricing] {canonical:<8} ({query}): NO SAMPLES")
        except Exception as e:
            print(f"[pricing] {canonical} failed: {type(e).__name__}: {e}")

    makeable, _unit, skipped = price_sushi(all_results)
    _cache = {name.lower(): info for name, info in makeable.items()}
    _skipped = skipped
    print(f"[pricing] priced {len(_cache)} sushi types, skipped {len(skipped)}")
    for name, reason in skipped:
        print(f"[pricing]   - {name}: {reason}")
    return True


def price_plate(counts: list[dict]) -> dict:
    if _cache is None:
        return {"available": False, "total": None, "breakdown": [], "location": LOCATION}

    breakdown: list[dict] = []
    total = 0.0
    for c in counts:
        key = c["display"].lower()
        info = _cache.get(key)
        qty = c["count"]
        if info is None:
            breakdown.append({
                "display": c["display"],
                "count": qty,
                "price_per_piece": None,
                "subtotal": None,
            })
            continue
        per_piece = info["price_per_piece"]
        subtotal = per_piece * qty
        breakdown.append({
            "display": c["display"],
            "count": qty,
            "price_per_piece": per_piece,
            "subtotal": subtotal,
        })
        total += subtotal

    return {
        "available": True,
        "total": round(total, 2),
        "breakdown": breakdown,
        "location": LOCATION,
    }
