# edgefinder_pro_singlefile_with_injuries_v2.py
import os
from datetime import datetime, timedelta
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EdgeFinder — Pro (with Injuries/Lineups, Sport Fix)", layout="wide")

def require_key(name):
    try:
        v = (st.secrets.get(name, "") or os.getenv(name, "")).strip()
    except Exception:
        v = (os.getenv(name, "")).strip()
    if not v:
        st.error(f"Missing {name}. Add it in Streamlit → Settings → Secrets.")
        st.stop()
    return v

ODDS_API_KEY  = require_key("ODDS_API_KEY")
APISPORTS_KEY = require_key("APISPORTS_KEY")

def odds_get(path, **params):
    url = "https://api.the-odds-api.com" + path
    r = requests.get(url, params={"apiKey": ODDS_API_KEY, **params}, timeout=25)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        st.error(f"Odds API error {r.status_code}: {err.get('message', r.text)[:300]}")
        st.stop()
    used = r.headers.get("x-requests-used")
    remaining = r.headers.get("x-requests-remaining")
    last = r.headers.get("x-requests-last")
    if used is not None:
        st.caption(f"Odds API usage: {{'X-Requests-Remaining': '{remaining}', 'X-Requests-Used': '{used}', 'X-Requests-Last': '{last}'}}")
    return r.json()

APISPORTS_BASE = {
    "Soccer":             "https://v3.football.api-sports.io",
    "Basketball":         "https://v1.basketball.api-sports.io",
    "Ice Hockey":         "https://v1.hockey.api-sports.io",
    "Baseball":           "https://v1.baseball.api-sports.io",
    "American Football":  "https://v1.american-football.api-sports.io",
}

def apisports_get(group: str, path: str, **params):
    base = APISPORTS_BASE.get(group)
    if not base:
        return {"response": []}
    headers = {"x-apisports-key": APISPORTS_KEY}
    r = requests.get(base + path, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        st.caption(f"API-SPORTS warn {r.status_code} {path}: {str(err)[:120]}")
        return {"response": []}
    return r.json()

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
    data = odds_get("/v4/sports")
    key_to_group = {d["key"]: (d.get("group") or group_from_key(d["key"])) for d in data}
    leagues = sorted([(d["key"], d.get("title","") or d["key"]) for d in data], key=lambda x: x[1])
    groups = sorted(set(key_to_group.values()) | {g for _, g in PREFIX_GROUPS} | {"Other"})
    return key_to_group, groups, leagues

key_to_group, SPORT_GROUPS, LEAGUES = sports_index()
DEFAULT_GROUP = "Basketball" if "Basketball" in SPORT_GROUPS else (SPORT_GROUPS[0] if SPORT_GROUPS else "Basketball")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_league(sport_key, regions="us", markets="h2h", odds_format="decimal"):
    return odds_get(f"/v4/sports/{sport_key}/odds", regions=regions, markets=markets, oddsFormat=odds_format)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upcoming(regions="us", markets="h2h", odds_format="decimal"):
    return odds_get("/v4/sports/upcoming/odds", regions=regions, markets=markets, oddsFormat=odds_format)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_scores(sport_key: str, days_from: int = 120):
    try:
        return odds_get(f"/v4/sports/{sport_key}/scores", daysFrom=days_from, dateFormat="iso")
    except Exception:
        return []

def fmt_time(iso):
    try:
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

def implied_from_decimal(dec_a, dec_b):
    try:
        a = 1.0 / float(dec_a); b = 1.0 / float(dec_b); s = a + b
        if s <= 0: return None, None
        return a / s, b / s
    except Exception:
        return None, None

def conf_meter(p):
    p = float(p or 0)
    blocks = max(0, min(10, int(round(p * 10))))
    return "■" * blocks + "□" * (10 - blocks)

def compute_last_n(team: str, scores: list, n: int = 10):
    games = [g for g in scores if g.get("completed") and team in (g.get("teams") or [g.get("home_team"), g.get("away_team")])]
    games.sort(key=lambda g: g.get("commence_time") or "", reverse=True)
    recent = games[:n]
    wins = 0; pf=0.0; pa=0.0
    for g in recent:
        h, a = g.get("home_team"), g.get("away_team")
        sc = g.get("scores") or []
        try:
            sdict = {x.get("name"): float(x.get("score")) for x in sc}
        except Exception:
            sdict = {}
        if sdict.get(h,0) > sdict.get(a,0): winner = h
        elif sdict.get(h,0) < sdict.get(a,0): winner = a
        else: winner = None
        if winner == team: wins += 1
        pf += sdict.get(team, 0.0)
        opp = a if team == h else h
        pa += sdict.get(opp, 0.0)
    gp = len(recent) if recent else 1
    return wins, len(recent), (pf - pa) / gp

def compute_h2h(team_a: str, team_b: str, scores: list, n: int = 6):
    games = [g for g in scores if g.get("completed")
             and team_a in (g.get("teams") or [g.get("home_team"), g.get("away_team")])
             and team_b in (g.get("teams") or [g.get("home_team"), g.get("away_team")])]
    games.sort(key=lambda g: g.get("commence_time") or "", reverse=True)
    games = games[:n]
    a_wins = 0
    for g in games:
        h, a = g.get("home_team"), g.get("away_team")
        sc = g.get("scores") or []
        try:
            sdict = {x.get("name"): float(x.get("score")) for x in sc}
        except Exception:
            sdict = {}
        winner = h if sdict.get(h, 0) > sdict.get(a, 0) else a
        if winner == team_a:
            a_wins += 1
    return a_wins, len(games)

def compute_elo(scores: list, k: float = 20.0, hfa: float = 30.0):
    teams = {}
    ordered = sorted([g for g in scores if g.get("completed")], key=lambda g: g.get("commence_time") or "")
    for g in ordered:
        h, a = g.get("home_team"), g.get("away_team")
        if not h or not a: continue
        teams.setdefault(h, 1500.0); teams.setdefault(a, 1500.0)
        sc = g.get("scores") or []
        try:
            sdict = {x.get("name"): float(x.get("score")) for x in sc}
            hs = sdict.get(h, 0.0); as_ = sdict.get(a, 0.0)
        except Exception:
            hs, as_ = 0.0, 0.0
        Rh, Ra = teams[h], teams[a]
        Eh = 1.0 / (1.0 + 10 ** ((Ra - (Rh + hfa)) / 400.0))
        Ea = 1.0 - Eh
        if hs > as_: Sh, Sa = 1.0, 0.0
        elif hs < as_: Sh, Sa = 0.0, 1.0
        else: Sh, Sa = 0.5, 0.5
        teams[h] = Rh + k * (Sh - Eh)
        teams[a] = Ra + k * (Sa - Ea)
    return teams

def rest_info(team: str, scores: list, as_of_iso: str):
    try:
        as_of = datetime.fromisoformat(as_of_iso.replace("Z", "+00:00"))
    except Exception:
        as_of = datetime.utcnow()
    games = [g for g in scores if g.get("completed") and team in (g.get("teams") or [g.get("home_team"), g.get("away_team")])]
    games.sort(key=lambda g: g.get("commence_time") or "", reverse=True)
    if not games:
        return 7, False
    last = games[0]
    try:
        last_time = datetime.fromisoformat(last.get("commence_time","").replace("Z","+00:00"))
    except Exception:
        last_time = as_of - timedelta(days=2)
    days = max(0, (as_of - last_time).days)
    b2b = (as_of - last_time) <= timedelta(hours=36)
    return days, b2b

def soccer_team_id(team_name: str):
    try:
        resp = apisports_get("Soccer", "/teams", search=team_name)
        arr = resp.get("response", [])
        if not arr: return None
        return (arr[0].get("team") or {}).get("id")
    except Exception:
        return None

def soccer_fixture_id(team_id: int, date_iso: str):
    try:
        ymd = (date_iso or "")[:10]
        fx = apisports_get("Soccer", "/fixtures", team=team_id, date=ymd).get("response", [])
        if not fx: return None
        for f in fx:
            stt = (f.get("fixture") or {}).get("status", {}) or {}
            if str(stt.get("short","")).upper() in {"NS","TBD"}:
                return (f.get("fixture") or {}).get("id")
        return (fx[0].get("fixture") or {}).get("id")
    except Exception:
        return None

def soccer_injuries_lineups(team_name: str, date_iso: str):
    tid = soccer_team_id(team_name)
    if not tid: return [], []
    fid = soccer_fixture_id(tid, date_iso)
    if not fid: return [], []
    inj = apisports_get("Soccer", "/injuries", fixture=fid).get("response", [])
    line = apisports_get("Soccer", "/fixtures/lineups", fixture=fid).get("response", [])
    inj_list = []
    for row in inj:
        p = row.get("player") or {}
        i = row.get("injury") or {}
        nm = p.get("name") or "Player"
        rs = i.get("reason") or i.get("type") or "Undisclosed"
        inj_list.append(f"{nm} — {rs}")
    starters = []
    pick = None
    for L in line:
        if (L.get("team") or {}).get("id") == tid:
            pick = L; break
    if not pick and line:
        pick = line[0]
    if pick:
        starters = [(p.get("player") or {}).get("name") for p in pick.get("startXI", []) if (p.get("player") or {}).get("name")]
    return inj_list, starters[:11]

def injuries_and_lineups(group: str, team_name: str, date_iso: str):
    if group == "Soccer":
        return soccer_injuries_lineups(team_name, date_iso)
    return [], []

def edgefinder_predict(home_team, away_team, sport_key, event, scores_cache=None, elo_table=None, only_book=None):
    group = group_from_key(sport_key)
    when_iso = event.get("commence_time") or ""

    best = best_ml_prices(event, only_book=only_book)
    dec_home = best.get(home_team, {}).get("price")
    dec_away = best.get(away_team, {}).get("price")
    mh = ma = None
    if dec_home is not None and dec_away is not None:
        mh, ma = implied_from_decimal(dec_home, dec_away)

    eh = ea = gap = None
    fh = fa = None
    hh = None
    rest_h = rest_a = 3
    b2b_h = b2b_a = False

    if scores_cache:
        if elo_table is None:
            elo_table = compute_elo(scores_cache)
        Rh = elo_table.get(home_team, 1500.0); Ra = elo_table.get(away_team, 1500.0)
        eh = 1.0 / (1.0 + 10 ** ((Ra - (Rh + 30.0)) / 400.0))
        ea = 1.0 - eh
        gap = Rh - Ra

        w_h, n_h, diff_h = compute_last_n(home_team, scores_cache, 10)
        w_a, n_a, diff_a = compute_last_n(away_team, scores_cache, 10)
        fh = (w_h / n_h) if n_h else None
        fa = (w_a / n_a) if n_a else None

        a_wins, n_h2h = compute_h2h(away_team, home_team, scores_cache, 6)
        if n_h2h:
            hh = 1.0 - (a_wins / n_h2h)

        as_of = when_iso
        rest_h, b2b_h = rest_info(home_team, scores_cache, as_of)
        rest_a, b2b_a = rest_info(away_team, scores_cache, as_of)

    inj_home, lineup_home = injuries_and_lineups(group, home_team, when_iso)
    inj_away, lineup_away = injuries_and_lineups(group, away_team, when_iso)

    weights, probs_home = [], []
    if mh is not None:               weights.append(0.45); probs_home.append(mh)
    if eh is not None:               weights.append(0.25); probs_home.append(eh)
    if fh is not None and fa is not None:
        form_prob = 0.5 + (fh - fa) * 0.12
        weights.append(0.15); probs_home.append(form_prob)
    if hh is not None:               weights.append(0.05); probs_home.append(hh)

    inj_adj = 0.0
    if inj_home or inj_away:
        if len(inj_home) >= 3 and len(inj_away) < 2: inj_adj -= 0.03
        if len(inj_away) >= 3 and len(inj_home) < 2: inj_adj += 0.03
    if lineup_home and not lineup_away: inj_adj += 0.02
    if lineup_away and not lineup_home: inj_adj -= 0.02

    rest_bump = 0.0
    if rest_h is not None and rest_a is not None:
        rest_bump += 0.02 * (rest_h - rest_a)
    if b2b_a and not b2b_h: rest_bump += 0.03
    if b2b_h and not b2b_a: rest_bump -= 0.03

    if weights:
        ph = sum(p*w for p, w in zip(probs_home, weights)) / sum(weights)
        ph = max(0.0, min(1.0, ph + max(-0.05, min(0.05, rest_bump + inj_adj))))
    else:
        ph = 0.55

    pick_team = home_team if ph >= 0.5 else away_team
    conf = ph if pick_team == home_team else (1.0 - ph)

    reasons = []
    if mh is not None:
        fav_price = dec_home if pick_team == home_team else dec_away
        dog_price = dec_away if pick_team == home_team else dec_home
        reasons.append(f"Market edge: fav {fav_price} vs dog {dog_price}")
    if eh is not None:
        reasons.append(f"ELO gap {gap:+.0f} (favours {'home' if gap>=0 else 'away'})")
    if fh is not None and fa is not None:
        reasons.append(f"Form last 10: {home_team} {int((fh or 0)*10)}-{10-int((fh or 0)*10)} | {away_team} {int((fa or 0)*10)}-{10-int((fa or 0)*10)}")
    if hh is not None:
        reasons.append(f"Head-to-head (recent) home share {hh:.0%}")
    tags = []
    if b2b_a and not b2b_h: tags.append("away on B2B")
    if rest_h - rest_a >= 2: tags.append(f"home +{rest_h-rest_a} rest days")
    if tags: reasons.append("Rest/Fatigue: " + ", ".join(tags))
    if inj_home: reasons.append(f"Injuries {home_team}: " + ", ".join(inj_home[:4]) + ("..." if len(inj_home)>4 else ""))
    if inj_away: reasons.append(f"Injuries {away_team}: " + ", ".join(inj_away[:4]) + ("..." if len(inj_away)>4 else ""))
    if lineup_home: reasons.append(f"Probable starters {home_team}: " + ", ".join(lineup_home[:5]) + ("..." if len(lineup_home)>5 else ""))
    if lineup_away: reasons.append(f"Probable starters {away_team}: " + ", ".join(lineup_away[:5]) + ("..." if len(lineup_away)>5 else ""))
    if pick_team == home_team:
        reasons.append(f"Home-court advantage favors {home_team}")
    else:
        reasons.append(f"{pick_team} favored despite travel")

    return {"pick_team": pick_team, "conf": float(conf), "reasons": reasons}

st.title("EdgeFinder — Pro (Injuries/Lineups + Sport Fix)")

regions_map = {"US":"us", "UK":"uk", "EU":"eu", "AU":"au"}
regions_sel = st.multiselect("Regions", ["US","UK","EU","AU"], default=["US"])
regions_param = ",".join(regions_map[x] for x in regions_sel) or "us"

col1, col2 = st.columns([1,1])
with col1:
    mode_by_sport = st.toggle("Run by sport (aggregate all leagues)", value=True)
with col2:
    only_dk = st.toggle("DraftKings only", value=True)

st.session_state["only_book"] = "DraftKings" if only_dk else None

if mode_by_sport:
    group = st.selectbox("Sport", SPORT_GROUPS, index=SPORT_GROUPS.index(DEFAULT_GROUP) if DEFAULT_GROUP in SPORT_GROUPS else 0)
else:
    league_names = [name for _, name in LEAGUES] or ["(No leagues loaded)"]
    league = st.selectbox("League", league_names)
    name_to_key = {name:key for key,name in LEAGUES}

if st.button("Refresh games", use_container_width=True):
    st.session_state.pop("events", None)

if "events" not in st.session_state:
    if mode_by_sport:
        sport_keys = [k for k,g in key_to_group.items() if g == group]
        merged = []
        for sk in sport_keys:
            try:
                merged.extend(fetch_league(sk, regions=regions_param, markets="h2h", odds_format="decimal") or [])
            except Exception:
                pass
        st.session_state["events"] = merged
    else:
        key = name_to_key.get(league, None) if not mode_by_sport else None
        st.session_state["events"] = fetch_league(key, regions=regions_param, markets="h2h", odds_format="decimal") if key else []

events = st.session_state.get("events", [])

scores_cache = []
elo_table = {}
if events:
    sk_first = events[0].get("sport_key","")
    scores_cache = fetch_scores(sk_first, days_from=120)
    elo_table = compute_elo(scores_cache)

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
        c1, c2 = st.columns([1,1])
        with c1:
            select_all = st.checkbox("Select all", value=(len(filtered) <= 10))
        selected = ids if select_all else st.multiselect("Select matchup(s)", ids, format_func=lambda i: labels[i])

        if selected and st.button("Run Predictions", type="primary", use_container_width=True):
            rows = []
            for i in selected:
                ev = filtered[i]
                home, away = parse_home_away(ev)
                pred = edgefinder_predict(
                    home, away, ev.get("sport_key",""), ev,
                    scores_cache=scores_cache, elo_table=elo_table,
                    only_book=("DraftKings" if only_dk else None)
                )
                pick_team = pred.get("pick_team", home)
                conf = float(pred.get("conf", 0.0))
                reasons = list(pred.get("reasons", []))

                best = best_ml_prices(ev, only_book=("DraftKings" if only_dk else None))
                dec_home = best.get(home, {}).get("price")
                dec_away = best.get(away, {}).get("price")

                when_txt = fmt_time(ev.get("commence_time"))
                odds_line = f"{home} {dec_home if dec_home is not None else '—'} | {away} {dec_away if dec_away is not None else '—'}"

                st.markdown(
                    f"🏀 **{away} vs {home}**\n"
                    f"───────────────────────────────────────\n"
                    f"📅 {when_txt}\n"
                    f"📈 **Odds (best price):** {odds_line}\n"
                    f"🤖 **Prediction:** **{pick_team}** favored to win ({(conf or 0):.0%})\n"
                    f"🟢 **Confidence Meter:** {conf_meter(conf or 0)}\n\n"
                    f"**💬 Detailed Reasoning:**"
                )
                for r in reasons[:10]:
                    st.markdown(f" • {r}")
                verdict = "Strong" if (conf or 0) >= 0.65 else ("Medium" if (conf or 0) >= 0.55 else "Lean")
                st.markdown(f"───────────────────────────────────────\n✅ **EdgeFinder Model Verdict:** {verdict} edge for **{pick_team}**")

                rows.append({
                    "When": when_txt,
                    "Sport": ev.get("sport_key"),
                    "Matchup": f"{away} @ {home}",
                    "Model Pick": pick_team,
                    "Shown Prob.": f"{(conf or 0):.0%}",
                    "Best Home ML": dec_home,
                    "Best Away ML": dec_away,
                })

            if rows:
                df = pd.DataFrame(rows)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV (summary)", data=csv,
                                   file_name="edgefinder_cards.csv", mime="text/csv")
