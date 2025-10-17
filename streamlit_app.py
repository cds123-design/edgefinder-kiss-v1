# streamlit_app.py — EdgeFinder Odds (stable UI: forms + session_state)
import os, requests
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EdgeFinder Odds", layout="wide")

# ---------------- API key ----------------
def get_api_key() -> str:
    key = ""
    try:
        key = (st.secrets.get("ODDS_API_KEY","") or os.getenv("ODDS_API_KEY","")).strip()
    except Exception:
        key = (os.getenv("ODDS_API_KEY","")).strip()
    if not key:
        st.error("Missing ODDS_API_KEY. Add it in Streamlit ➜ Settings ➜ Secrets.")
        st.stop()
    return key

ODDS_KEY = get_api_key()

# ---------------- HTTP helper ----------------
def odds_get(path: str, **params):
    url = f"https://api.the-odds-api.com{path}"
    r = requests.get(url, params={"apiKey": ODDS_KEY, **params}, timeout=25)
    usage = {k:v for k,v in r.headers.items() if k.lower().startswith("x-requests")}
    if usage: st.caption(f"Odds API usage: {usage}")
    if r.status_code != 200:
        try: err = r.json()
        except Exception: err = {"message": r.text}
        if err.get("error_code") == "OUT_OF_USAGE_CREDITS":
            st.error("The Odds API: out of usage credits. Upgrade/add credits or try later.")
        else:
            st.error(f"Odds API error {r.status_code}: {err.get('message', r.text)[:300]}")
        st.stop()
    return r.json()

# ---------------- Sports dropdown ----------------
FALLBACK_SPORTS = {
    "basketball_nba": "Basketball — NBA",
    "basketball_euroleague": "Basketball — EuroLeague",
    "icehockey_nhl": "Hockey — NHL",
    "baseball_mlb": "Baseball — MLB",
    "americanfootball_nfl": "Football — NFL",
    "soccer_epl": "Soccer — England: Premier League",
    "soccer_italy_serie_a": "Soccer — Italy: Serie A",
    "soccer_spain_la_liga": "Soccer — Spain: La Liga",
    "soccer_uefa_champs_league": "Soccer — UEFA Champions League",
}

@st.cache_data(ttl=12*60*60, show_spinner=False)
def get_sports_map():
    try:
        data = odds_get("/v4/sports")
        items = sorted(data, key=lambda d: (d.get("group",""), d.get("title","")))
        return {d["key"]: f'{d.get("group","")} — {d.get("title","")} ({d["key"]})' for d in items}
    except Exception:
        return FALLBACK_SPORTS

sports_map = get_sports_map()
default_key = "basketball_nba" if "basketball_nba" in sports_map else next(iter(sports_map))

# ---------------- Cached fetchers ----------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_sport_events(sport_key: str, regions="us", markets="h2h", odds_format="decimal"):
    return odds_get("/v4/sports/{}/odds".format(sport_key),
                    regions=regions, markets=markets, oddsFormat=odds_format)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upcoming(regions="us", markets="h2h", odds_format="decimal"):
    return odds_get("/v4/sports/upcoming/odds",
                    regions=regions, markets=markets, oddsFormat=odds_format)

# ---------------- Utils ----------------
def fmt_time(iso: str) -> str:
    try:
        t = datetime.fromisoformat(iso.replace("Z","+00:00")).astimezone()
        return t.strftime("%b %d, %I:%M %p")
    except Exception:
        return iso or "TBD"

def event_id(ev: dict) -> str:
    if ev.get("id"): return ev["id"]
    h, a, ts = ev.get("home_team",""), ev.get("away_team",""), ev.get("commence_time",""),
    return f"{h}|{a}|{ts}"

def parse_home_away(ev: dict):
    home = ev.get("home_team", "TBD")
    away = ev.get("away_team")
    if not away:
        teams = ev.get("teams", [])
        if home and teams:
            away = next((t for t in teams if t != home), teams[0] if teams else "TBD")
        else:
            away = "TBD"
    return home, away

def label_for_event(ev: dict):
    home, away = parse_home_away(ev)
    return f"{away} @ {home} — {fmt_time(ev.get('commence_time','TBD'))}"

def best_ml_prices(ev: dict):
    best = {}
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h": continue
            for o in m.get("outcomes", []):
                team, price = o.get("name"), o.get("price")
                if team is None or price is None: continue
                if team not in best or price > best[team]["price"]:
                    best[team] = {"price": price, "bookmaker": bk.get("title")}
    return best

# ---------------- Model hook ----------------
try:
    from edgefinder import edgefinder_predict
except Exception:
    def edgefinder_predict(home_team: str, away_team: str, sport_key: str, event: dict):
        best = best_ml_prices(event)
        pick_team = max(best, key=lambda t: best[t]['price']) if best else home_team
        return {"pick_team": pick_team, "conf": 0.55}

# ---------------- UI: Step 1 (Get games) ----------------
st.title("EdgeFinder — Odds & Predictions")

with st.form("fetch_form", clear_on_submit=False):
    col1, col2 = st.columns([1,1])
    with col1:
        use_upcoming = st.toggle("Use 'All upcoming' (many leagues in one call)", value=False)
    with col2:
        odds_format = st.selectbox("Odds format", ["decimal","american","fractional"], index=0)

    col3, col4, col5 = st.columns([1,1,1])
    with col3:
        if not use_upcoming:
            sport_key = st.selectbox(
                "Sport",
                options=list(sports_map.keys()),
                index=list(sports_map.keys()).index(default_key),
                format_func=lambda k: sports_map[k]
            )
        else:
            sport_key = None
    with col4:
        regions = st.text_input("Regions", "us")
    with col5:
        markets = st.text_input("Markets", "h2h")

    force_refresh = st.checkbox("Force refresh (ignore cache)", value=False)
    submitted = st.form_submit_button("Get games")

if submitted:
    if force_refresh:
        fetch_sport_events.clear(); fetch_upcoming.clear()
    if use_upcoming:
        events = fetch_upcoming(regions, markets, odds_format)
    else:
        events = fetch_sport_events(sport_key, regions, markets, odds_format)
    st.session_state["events"] = events
    st.session_state["fetch_params"] = {
        "use_upcoming": use_upcoming,
        "sport_key": sport_key,
        "regions": regions,
        "markets": markets,
        "odds_format": odds_format
    }
    st.success(f"Loaded {len(events)} events.")

# ---------------- UI: Step 2 (Pick matchups & Predict) ----------------
if "events" in st.session_state and st.session_state["events"]:
    events = st.session_state["events"]
    params = st.session_state.get("fetch_params", {})
    live_sport_key = params.get("sport_key")  # may be None with 'upcoming'

    q = st.text_input("Filter by team (optional)")
    filtered = [e for e in events if not q or q.lower() in label_for_event(e).lower()]
    if not filtered:
        st.info("No events match the filter.")
    else:
        id_to_event = {event_id(e): e for e in filtered}
        id_to_label = {eid: label_for_event(e) for eid, e in id_to_event.items()}

        default_sel = st.session_state.get("selected_ids") or list(id_to_event.keys())[:1]
        selected_ids = st.multiselect(
            "Select matchup(s)",
            options=list(id_to_event.keys()),
            default=default_sel,
            format_func=lambda eid: id_to_label[eid]
        )
        st.session_state["selected_ids"] = selected_ids

        with st.form("predict_form"):
            manual = st.toggle("Enable manual override (only when selecting ONE game)", value=False)
            run = st.form_submit_button("Run Predictions")

        if run and selected_ids:
            rows = []
            for eid in selected_ids:
                ev = id_to_event[eid]
                home, away = parse_home_away(ev)

                if manual and len(selected_ids) == 1:
                    home = st.text_input("Home team", value=home, key=f"home_{eid}")
                    away = st.text_input("Away team", value=away, key=f"away_{eid}")

                best = best_ml_prices(ev)
                best_home = best.get(home, {})
                best_away = best.get(away, {})

                sk = ev.get("sport_key", live_sport_key)
                pred = edgefinder_predict(home, away, sk, ev)
                pick_team = pred.get("pick_team", home)
                conf = pred.get("conf", 0.0)

                rows.append({
                    "When": fmt_time(ev.get("commence_time")),
                    "Sport": sk,
                    "Matchup": f"{away} @ {home}",
                    "Model Pick": pick_team,
                    "Model Conf.": f"{conf:.0%}",
                    "Best Home ML": best_home.get("price"),
                    "Home BK": best_home.get("bookmaker"),
                    "Best Away ML": best_away.get("price"),
                    "Away BK": best_away.get("bookmaker"),
                })

            df = pd.DataFrame(rows)
            try:
                df["_c"] = df["Model Conf."].str.rstrip("%").astype(float)
                df = df.sort_values("_c", ascending=False).drop(columns=["_c"])
            except Exception:
                pass

            st.success(f"{len(df)} prediction(s) ready")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="edgefinder_predictions.csv", mime="text/csv")


st.caption("Tip: Use the 'All upcoming' toggle to batch leagues in one call and save credits.")