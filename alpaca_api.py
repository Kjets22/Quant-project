"""
alpaca_api.py — thin REST client for the Alpaca PAPER trading API.

Reads ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL from .env (git-ignored).
Plain `requests` — no SDK dependency. Paper keys (PK...) cannot touch real money.
"""

from __future__ import annotations

from pathlib import Path

from data import enable_truststore

enable_truststore()

import requests  # noqa: E402


def _load_env():
    env = {}
    p = Path(__file__).with_name(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


_E = _load_env()
BASE = _E.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
HDRS = {"APCA-API-KEY-ID": _E.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": _E.get("ALPACA_SECRET_KEY", "")}

assert "paper" in BASE, "SAFETY: this client only talks to the PAPER endpoint."


def _req(method, path, **kw):
    # GETs retry once on timeout/connection drops (Alpaca has flaky nights);
    # POST/PATCH/DELETE never auto-retry (duplicate-order risk).
    last = None
    for attempt in (1, 2):
        try:
            r = requests.request(method, f"{BASE}{path}", headers=HDRS,
                                 timeout=60, **kw)
            break
        except requests.exceptions.RequestException as e:
            last = e
            if method != "GET" or attempt == 2:
                raise
    else:
        raise last
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json() if r.text else {}


def account():
    return _req("GET", "/v2/account")


def clock():
    return _req("GET", "/v2/clock")


def positions():
    return _req("GET", "/v2/positions")


def position(symbol):
    try:
        return _req("GET", f"/v2/positions/{symbol}")
    except RuntimeError:
        return None


def open_orders():
    return _req("GET", "/v2/orders", params={"status": "open", "limit": 500})


def closed_orders(after_iso, symbols=None):
    p = {"status": "closed", "limit": 500, "after": after_iso, "direction": "desc"}
    if symbols:
        p["symbols"] = ",".join(symbols)
    return _req("GET", "/v2/orders", params=p)


def submit_bracket(symbol, qty, take_profit, stop_loss, client_id):
    """Market BUY + bracket (sell-limit target / sell-stop). Whole shares, GTC legs."""
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "market",
        "time_in_force": "day", "order_class": "bracket",
        "client_order_id": client_id,
        "take_profit": {"limit_price": str(round(take_profit, 2))},
        "stop_loss": {"stop_price": str(round(stop_loss, 2))},
    })


def get_order(order_id):
    """Full order state incl. bracket legs (nested)."""
    return _req("GET", f"/v2/orders/{order_id}", params={"nested": "true"})


def submit_bracket_short(symbol, qty, take_profit, stop_loss, client_id):
    """Market SELL-SHORT + bracket (buy-limit target below / buy-stop above)."""
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "sell", "type": "market",
        "time_in_force": "day", "order_class": "bracket",
        "client_order_id": client_id,
        "take_profit": {"limit_price": str(round(take_profit, 2))},
        "stop_loss": {"stop_price": str(round(stop_loss, 2))},
    })


def submit_limit_bracket_short(symbol, qty, limit_price, take_profit, stop_loss,
                               client_id):
    """LIMIT SELL-SHORT at the signal price + bracket."""
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "sell", "type": "limit",
        "limit_price": str(round(limit_price, 2)), "time_in_force": "day",
        "order_class": "bracket", "client_order_id": client_id,
        "take_profit": {"limit_price": str(round(take_profit, 2))},
        "stop_loss": {"stop_price": str(round(stop_loss, 2))},
    })


def submit_limit_bracket(symbol, qty, limit_price, take_profit, stop_loss, client_id):
    """LIMIT BUY at the signal price + bracket. Day order: dies at close if unfilled."""
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "limit",
        "limit_price": str(round(limit_price, 2)), "time_in_force": "day",
        "order_class": "bracket", "client_order_id": client_id,
        "take_profit": {"limit_price": str(round(take_profit, 2))},
        "stop_loss": {"stop_price": str(round(stop_loss, 2))},
    })


def market_sell(symbol, qty, client_id):
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "sell", "type": "market",
        "time_in_force": "day", "client_order_id": client_id})


def submit_limit(symbol, qty, limit_price, client_id, extended=False):
    """Plain DAY limit BUY; extended=True makes it eligible 4:00-20:00 ET.
    (Alpaca allows ONLY day limit orders in extended hours — no market/bracket.)"""
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "limit",
        "limit_price": str(round(limit_price, 2)), "time_in_force": "day",
        "extended_hours": bool(extended), "client_order_id": client_id})


def limit_sell(symbol, qty, limit_price, client_id, extended=False):
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "sell", "type": "limit",
        "limit_price": str(round(limit_price, 2)), "time_in_force": "day",
        "extended_hours": bool(extended), "client_order_id": client_id})


def market_buy(symbol, qty, client_id):
    """Plain market BUY (works for stock and OCC option symbols alike)."""
    return _req("POST", "/v2/orders", json={
        "symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "market",
        "time_in_force": "day", "client_order_id": client_id})


def option_contracts(underlying, exp_gte, exp_lte, k_lo, k_hi, ctype="call"):
    """Tradable option contracts on the PAPER account (paginated)."""
    out, token = [], None
    while True:
        p = {"underlying_symbols": underlying, "type": ctype, "limit": 500,
             "expiration_date_gte": exp_gte, "expiration_date_lte": exp_lte,
             "strike_price_gte": str(k_lo), "strike_price_lte": str(k_hi)}
        if token:
            p["page_token"] = token
        j = _req("GET", "/v2/options/contracts", params=p)
        out += j.get("option_contracts", [])
        token = j.get("next_page_token")
        if not token:
            return out


def cancel_order(order_id):
    r = requests.delete(f"{BASE}/v2/orders/{order_id}", headers=HDRS, timeout=30)
    return r.status_code in (200, 204, 404)


def cancel_symbol_orders(symbol):
    n = 0
    for o in open_orders():
        if o["symbol"] == symbol and cancel_order(o["id"]):
            n += 1
    return n


def close_position(symbol):
    r = requests.delete(f"{BASE}/v2/positions/{symbol}", headers=HDRS, timeout=30)
    return r.status_code in (200, 204, 404)


if __name__ == "__main__":
    import json
    a = account()
    print("ACCOUNT OK:", a["status"], " equity=$", a["equity"], " cash=$", a["cash"])
    c = clock()
    print("CLOCK:", "OPEN" if c["is_open"] else "closed", "| next open:", c["next_open"])
    print("positions:", json.dumps(positions()))
    print("open orders:", len(open_orders()))
