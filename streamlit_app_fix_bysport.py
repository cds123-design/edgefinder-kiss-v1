# streamlit_app_fix_bysport.py — EdgeFinder (By-sport fix + Regions)
import os
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EdgeFinder — By Sport Fix", layout="wide")

# ----------------------------- Secrets / Key -----------------------------
def get_api_key():
    try:
        key = (st.secrets.get("ODDS_API_KEY", "") or os.getenv("ODDS_API_KEY", "")).strip()
    except Exception:
        key = (os.getenv("ODDS_API_KEY", "")).strip()
    if not key:
        st.error("Missing ODDS_API_KEY. Add it in Streamlit → Settings → Secrets.")
        st.stop()
    return key

ODDS_KEY = get_api_key()

# ----------------------------- HTTP helper -------------------------------
def odds_get(path, **params):
    url = "https://api.the-odds-api.com" + path
    r = requests.get(url, params={"apiKey": ODDS_KEY, **params}, timeout=25)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        st.error(f"Odds API error {r.status_code}: {err.get('message', r.text)[:300]}")
        st.stop()
    # show usage headers if present
    used = r.headers.get("x-requests-used")
    remaining = r.headers.get("x-requests-remaining")
    last = r.headers.get("x-requests-last")
    if used is not None:
        st.caption(f"Odds API usage: {{'X-Requests-Used': '{used}', 'X-Requests-Remaining': '{remaining}', 'X-Requests-Last': '{last}'}}")
    return r.json()

# ----------------------------- Grouping ----------------------------------
PREFIX_GROUPS = [
    ("basketball_", "Basketball"),
    ("americanfootball_", "American Football"),
    ("icehockey_", "Ice Hockey"),
    ("baseball_", "Baseball"),
    ("soccer_", "Soccer"),
]

def group_from_key(key: str) -> str:
    k = key or ""
    for pref, grp in PREFIX_GROUPS:
        if k.startswith(pref):
            return grp
    return "Other"

@st.cache_data(ttl=12*60*60, show_spinner=False)
def sports_index():
    try:
        data = odds_get("/v4/sports")
        key_to_group = {d["key"]: (d.get("group") or group_from_key(d["key"])) for d in data}
        leagues = sorted([(d["key"], d.get("title","") or d["key"]) for d in data], key=lambda x: x[1])
        groups = sorted(set(key_to_group.values()) | {g for _, g in PREFIX_GROUPS} | {"Other"})
        return key_to_group, groups, leagues
    except Exception:
        return {}, [g for _, g in PREFIX_GROUPS] + ["Other"], []

key_to_group, SPORT_GROUPS, LEAGUES = sports_index()
DEFAULT_GROUP = "Basketball" if "Basketball" in SPORT_GROUPS else (SPORT_GROUPS[0] if SPORT_GROUPS else "Basketball")

# ----------------------------- Fetchers ----------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_upcoming(regions="us", markets="h2h", odds_format="decimal"):
    return odds_get("/v4/sports/upcoming/odds", regions=regions, markets=markets, oddsFormat=odds_format)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_league(sport_key, regions="us", markets="h2h", odds_format="decimal"):
    return odds_get(f"/v4/sports/{sport_key}/odds", regions=regions, markets=markets, oddsFormat=odds_format)

# ----------------------------- Utilities ---------------------------------
def fmt_time(iso):
    try:
        from datetime import timezone
        t = datetime.fromisoformat((iso or "").replace("Z", "+00:00")).astimezone()
        return t.strftime("%b %d, %I:%M %p")
    except Exception:
        return iso or "TBD"

def parse_home_away(ev):
    home = ev.get("home_team", "TBD")
    away = ev.get("away_team")
    if not away:
        teams = ev.get("teams", [])
        if home and teams:
            away = next((t for t in teams if t != home), teams[0] if teams else "TBD")
        else:
            away = "TBD"
    return home, away

def label_for_event(ev):
    home, away = parse_home_away(ev)
    return f"{away} @ {home} — {fmt_time(ev.get('commence_time','TBD'))}"

def best_ml_prices(ev, only_book=None):
    best = {}
    for bk in ev.get("bookmakers", []):
        title = (bk.get("title") or "").strip()
        if only_book and title.lower() != str(only_book).lower():
            continue
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []):
                team, price = o.get("name"), o.get("price")
                if team is None or price is None:
                    continue
                if team not in best or price > best[team]["price"]:
                    best[team] = {"price": price, "bookmaker": title or only_book or "Best"}
    return best

def conf_meter(p):
    p = float(p or 0)
    blocks = max(0, min(10, int(round(p * 10))))
    return "■" * blocks + "□" * (10 - blocks)

def implied_from_decimal(dec_a, dec_b):
    try:
        a = 1.0 / float(dec_a); b = 1.0 / float(dec_b); s = a + b
        if s <= 0: return None, None
        return a / s, b / s
    except Exception:
        return None, None

# ----------------------------- Fallback Model ----------------------------
def edgefinder_predict(home_team, away_team, sport_key, event):
    only_book = st.session_state.get("only_book")
    best = best_ml_prices(event, only_book=only_book)
    dec_home = best.get(home_team, {}).get("price")
    dec_away = best.get(away_team, {}).get("price")
    if dec_home is not None and dec_away is not None:
        ph, pa = implied_from_decimal(dec_home, dec_away)
        pick_team = home_team if ph >= pa else away_team
        conf = max(ph, pa)
    else:
        pick_team = home_team; conf = 0.55
    return {"pick_team": pick_team, "conf": float(conf), "reasons": []}

# ----------------------------- UI ---------------------------------------
st.title("EdgeFinder — Run by Sport")

mode = st.toggle("Run by sport (All Upcoming)", value=True)
only_dk = st.toggle("DraftKings only", value=True)
st.session_state["only_book"] = "DraftKings" if only_dk else None

# Regions control (fix for books outside US/Euroleague issues)
regions_map = {"US":"us", "UK":"uk", "EU":"eu", "AU":"au"}
regions_sel = st.multiselect("Regions", ["US","UK","EU","AU"], default=["US"])
regions_param = ",".join(regions_map[x] for x in regions_sel) or "us"

st.button("Refresh games")

if mode:
    # BY SPORT (fixed): aggregate ALL leagues in the selected group
    group = st.selectbox("Sport", SPORT_GROUPS, index=SPORT_GROUPS.index(DEFAULT_GROUP) if DEFAULT_GROUP in SPORT_GROUPS else 0)
else:
    league_names = [name for _, name in LEAGUES] or ["(No leagues loaded)"]
    league = st.selectbox("League", league_names)
    name_to_key = {name:key for key,name in LEAGUES}

# Fetch events
if mode:
    # collect all sport_keys in this group and fetch each league, merging
    sport_keys = [k for k,g in key_to_group.items() if g == group]
    merged = []
    for sk in sport_keys:
        try:
            evs = fetch_league(sk, regions=regions_param, markets="h2h", odds_format="decimal")
            merged.extend(evs or [])
        except Exception:
            pass
    events = merged
else:
    key = name_to_key.get(league)
    events = fetch_league(key, regions=regions_param, markets="h2h", odds_format="decimal") if key else []

if not events:
    st.info("No upcoming events for the current filters.")
else:
    q = st.text_input("Filter by team (optional)")
    filtered = [e for e in events if not q or q.lower() in label_for_event(e).lower()]
    if not filtered:
        st.info("No events match the filter.")
    else:
        labels = [label_for_event(e) for e in filtered]
        ids = list(range(len(filtered)))
        select_all = st.checkbox("Select all", value=(len(filtered) <= 10))
        selected = ids if select_all else st.multiselect("Select matchup(s)", ids, format_func=lambda i: labels[i])
        if selected and st.button("Run Predictions", type="primary", use_container_width=True):
            rows = []
            for i in selected:
                ev = filtered[i]
                home = ev.get("home_team"); away = ev.get("away_team")
                pred = edgefinder_predict(home, away, ev.get("sport_key",""), ev)
                pick_team = pred["pick_team"]; conf = pred["conf"]
                best = best_ml_prices(ev, only_book=("DraftKings" if only_dk else None))
                dec_home = best.get(home, {}).get("price"); dec_away = best.get(away, {}).get("price")
                when_txt = fmt_time(ev.get("commence_time"))
                odds_line = f"{home} {dec_home if dec_home is not None else '—'} | {away} {dec_away if dec_away is not None else '—'}"
                st.markdown(
                    f"🏀 **{away} vs {home}**\n"
                    f"───────────────────────────────────────\n"
                    f"📅 {when_txt}\n"
                    f"📈 **Odds (best price):** {odds_line}\n"
                    f"🤖 **Prediction:** **{pick_team}** favored to win ({conf:.0%})\n"
                    f"🟢 **Confidence Meter:** {conf_meter(conf)}\n"
                )
                rows.append({"When": when_txt, "Sport": ev.get("sport_key"), "Matchup": f\"{away} @ {home}\", "Pick": pick_team, "Prob": f\"{conf:.0%}\"})
            if rows:
                df = pd.DataFrame(rows)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="edgefinder_cards.csv", mime="text/csv")
