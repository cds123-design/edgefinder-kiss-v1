
# streamlit_app.py — EdgeFinder Pro
# - NBA/NHL/NFL/MLB injuries via SportsDataIO Players?injured=true (strict team filtering, hide masked entries)
# - Soccer injuries via API-FOOTBALL with robust fallback (fixture by date -> next fixture -> league+season+team)
import os
from datetime import datetime, timedelta
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EdgeFinder — Pro", layout="wide")

def get_secret(name, required=False):
    try:
        v = (st.secrets.get(name, "") or os.getenv(name, "")).strip()
    except Exception:
        v = (os.getenv(name, "")).strip()
    if required and not v:
        st.error(f"Missing {name}. Add it in Streamlit → Settings → Secrets.")
        st.stop()
    return v

ODDS_API_KEY       = get_secret("ODDS_API_KEY", required=True)
APISPORTS_KEY      = get_secret("APISPORTS_KEY", required=True)
SPORTSDATA_API_KEY = get_secret("SPORTSDATA_API_KEY", required=False)

# ========================= Odds API helpers =========================
def odds_get(path, **params):
    url = "https://api.the-odds-api.com" + path
    params = {k:v for k,v in params.items() if v not in (None, "", [])}
    r = requests.get(url, params={"apiKey": ODDS_API_KEY, **params}, timeout=25)
    if r.status_code == 200:
        used = r.headers.get("x-requests-used")
        remaining = r.headers.get("x-requests-remaining")
        last = r.headers.get("x-requests-last")
        if used is not None:
            st.caption(f"Odds API usage: {{'X-Requests-Remaining': '{remaining}', 'X-Requests-Used': '{used}', 'X-Requests-Last': '{last}'}}")
        return r.json()
    try:
        msg = r.json().get("message", r.text)
    except Exception:
        msg = r.text
    if r.status_code in (400,401,403,404,409,422):
        st.caption(f"Odds API {r.status_code} {path}: {str(msg)[:160]} (continuing)")
        return []
    st.error(f"Odds API error {r.status_code}: {str(msg)[:300]}"); st.stop()

# ========================= API-FOOTBALL (Soccer) =========================
APIS_FOOTBALL_BASE = "https://v3.football.api-sports.io"
def apif_get(path: str, **params):
    headers = {"x-apisports-key": APISPORTS_KEY}
    r = requests.get(APIS_FOOTBALL_BASE + path, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return {"response": []}
    return r.json()

COUNTRY_FIX = {"usa": "USA", "unitedstates": "USA", "england": "England",
               "scotland":"Scotland","wales":"Wales","northernireland":"Northern Ireland",
               "southkorea":"Korea Republic","ivorycoast":"Cote D'Ivoire"}

def soccer_country_from_key(sport_key: str) -> str|None:
    if not (sport_key or "").startswith("soccer_"): return None
    parts = (sport_key or "").split("_")
    if len(parts) < 2: return None
    c_raw = parts[1].replace("-", "").lower()
    return COUNTRY_FIX.get(c_raw, parts[1].replace("-", " ").title())

def soccer_team_id(team_name: str, country_hint: str|None):
    params = {"search": team_name}
    if country_hint: params["country"] = country_hint
    resp = apif_get("/teams", **params).get("response", [])
    if not resp and country_hint:
        resp = apif_get("/teams", search=team_name).get("response", [])
    if not resp: return None
    # pick best match by simple containment
    tgt = (team_name or "").lower()
    best_id, best_score = None, -1
    for row in resp:
        nm = ((row.get("team") or {}).get("name") or "").lower()
        score = 100 if nm == tgt else (len(set(nm) & set(tgt)))
        if score > best_score:
            best_score = score; best_id = (row.get("team") or {}).get("id")
    return best_id

def soccer_fixture_by_date_or_next(team_id: int, date_iso: str):
    # 1) by date (API requires yyyy-mm-dd in local timezone)
    ymd = (date_iso or "")[:10] or datetime.utcnow().strftime("%Y-%m-%d")
    fx = apif_get("/fixtures", team=team_id, date=ymd).get("response", [])
    # 2) fallback: next fixture for team
    if not fx:
        fx = apif_get("/fixtures", team=team_id, next=1).get("response", [])
    if not fx: return None
    # prefer NS/TBD (not started) fixtures
    fx_sorted = sorted(fx, key=lambda f: (0 if ((f.get("fixture") or {}).get("status", {}) or {}).get("short") in ("NS","TBD") else 1))
    return fx_sorted[0]

def soccer_injuries_and_lineups(team_name: str, date_iso: str, sport_key: str):
    country = soccer_country_from_key(sport_key)
    tid = soccer_team_id(team_name, country)
    if not tid: return [], []
    fx = soccer_fixture_by_date_or_next(tid, date_iso)
    injuries, starters = [], []
    league_id, season = None, None
    if fx:
        fixture_id = (fx.get("fixture") or {}).get("id")
        league_id = (fx.get("league") or {}).get("id")
        season    = (fx.get("league") or {}).get("season")
        if fixture_id:
            inj = apif_get("/injuries", fixture=fixture_id).get("response", [])
            for row in inj:
                p = row.get("player") or {}; i = row.get("injury") or {}
                nm = p.get("name") or "Player"; rs = i.get("reason") or i.get("type") or "Undisclosed"
                injuries.append(f"{nm} — {rs}")
            line = apif_get("/fixtures/lineups", fixture=fixture_id).get("response", [])
            # choose matching team
            pick = None
            for L in line:
                if (L.get("team") or {}).get("id") == tid:
                    pick = L; break
            if not pick and line: pick = line[0]
            if pick:
                starters = [(p.get("player") or {}).get("name") for p in pick.get("startXI", []) if (p.get("player") or {}).get("name")]
    # Fallback: league+season+team injuries
    if not injuries and league_id and season:
        inj = apif_get("/injuries", league=league_id, season=season, team=tid).get("response", [])
        for row in inj:
            p = row.get("player") or {}; i = row.get("injury") or {}
            nm = p.get("name") or "Player"; rs = i.get("reason") or i.get("type") or "Undisclosed"
            injuries.append(f"{nm} — {rs}")
    # cap size
    return injuries[:10], starters[:11]

# ========================= SportsDataIO (US leagues) =========================
SDIO_SPORT_FOR_KEY = {"basketball_nba":"nba","icehockey_nhl":"nhl","baseball_mlb":"mlb","americanfootball_nfl":"nfl"}
def sdio_get(sport_slug: str, endpoint: str, **params):
    if not SPORTSDATA_API_KEY or not sport_slug: return None
    base = f"https://api.sportsdata.io/v3/{sport_slug}/scores/json"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(f"{base}/{endpoint}", headers=headers, params=params, timeout=25)
    if r.status_code != 200: return None
    try: return r.json()
    except Exception: return None

@st.cache_data(ttl=300, show_spinner=False)
def sdio_players_injured(sport_slug: str):
    return sdio_get(sport_slug, "Players", injured="true") or []

NBA_TEAM_KEY = {
    "Atlanta Hawks":"ATL","Boston Celtics":"BOS","Brooklyn Nets":"BKN","Charlotte Hornets":"CHA",
    "Chicago Bulls":"CHI","Cleveland Cavaliers":"CLE","Dallas Mavericks":"DAL","Denver Nuggets":"DEN",
    "Detroit Pistons":"DET","Golden State Warriors":"GSW","Houston Rockets":"HOU","Indiana Pacers":"IND",
    "Los Angeles Clippers":"LAC","Los Angeles Lakers":"LAL","Memphis Grizzlies":"MEM","Miami Heat":"MIA",
    "Milwaukee Bucks":"MIL","Minnesota Timberwolves":"MIN","New Orleans Pelicans":"NOP","New York Knicks":"NYK",
    "Oklahoma City Thunder":"OKC","Orlando Magic":"ORL","Philadelphia 76ers":"PHI","Phoenix Suns":"PHO",
    "Portland Trail Blazers":"POR","Sacramento Kings":"SAC","San Antonio Spurs":"SAS","Toronto Raptors":"TOR",
    "Utah Jazz":"UTA","Washington Wizards":"WAS"
}

def cleaned_parts(status, body, notes):
    def c(x):
        if not x: return ""
        s = str(x).strip()
        return "" if s.lower()=="scrambled" else s
    parts = [c(status), c(body), c(notes)]
    return [p for p in parts if p]

def sdio_injuries_for_team(sport_key: str, team_name: str):
    sport_slug = SDIO_SPORT_FOR_KEY.get(sport_key)
    if not sport_slug: return []
    team_key = NBA_TEAM_KEY.get(team_name) if sport_key=="basketball_nba" else None
    players = sdio_players_injured(sport_slug) or []
    out = []
    for p in players:
        # strict filter
        if sport_key=="basketball_nba":
            if team_key is None or str(p.get("Team")) != team_key: continue
        else:
            if str(p.get("TeamName","")).lower() != team_name.lower() and str(p.get("Team")).lower() not in team_name.lower():
                continue
        parts = cleaned_parts(p.get("InjuryStatus") or p.get("Status"),
                              p.get("InjuryBodyPart") or p.get("BodyPart"),
                              p.get("InjuryNotes") or p.get("Notes"))
        if not parts:  # masked entries - skip
            continue
        nm = f"{p.get('FirstName','')} {p.get('LastName','')}".strip() or "Player"
        out.append(f"{nm} — " + " — ".join(parts))
    # dedup + cap
    seen=set(); uniq=[]
    for s in out:
        if s not in seen: uniq.append(s); seen.add(s)
    return uniq[:8]

# ========================= Grouping & fetchers =========================
PREFIX_GROUPS=[("basketball_","Basketball"),("americanfootball_","American Football"),
               ("icehockey_","Ice Hockey"),("baseball_","Baseball"),("soccer_","Soccer")]
def group_from_key(key: str) -> str:
    for pref,g in PREFIX_GROUPS:
        if (key or "").startswith(pref): return g
    return "Other"

@st.cache_data(ttl=12*60*60, show_spinner=False)
def sports_index():
    data = odds_get("/v4/sports")
    if not isinstance(data, list): data=[]
    key_to_group = {d.get("key",""): (d.get("group") or group_from_key(d.get("key",""))) for d in data if d.get("key")}
    leagues = sorted([(d.get("key",""), d.get("title","") or d.get("key","")) for d in data if d.get("key")], key=lambda x:x[1])
    groups  = sorted(set(key_to_group.values()) | {g for _,g in PREFIX_GROUPS} | {"Other"})
    return key_to_group, groups, leagues

key_to_group, SPORT_GROUPS, LEAGUES = sports_index()
DEFAULT_GROUP = "Basketball" if "Basketball" in SPORT_GROUPS else (SPORT_GROUPS[0] if SPORT_GROUPS else "Basketball")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_league_robust(sport_key, regions="us", markets_try=("h2h","spreads","totals")):
    out=[]
    for m in markets_try:
        data = odds_get(f"/v4/sports/{sport_key}/odds", regions=regions, markets=m, oddsFormat="decimal")
        if isinstance(data, list) and data:
            out.extend(data); break
    if not out:
        data = odds_get(f"/v4/sports/{sport_key}/odds", regions=regions, oddsFormat="decimal")
        if isinstance(data, list) and data: out.extend(data)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def fetch_scores(sport_key: str, days_from: int = 120):
    data = odds_get(f"/v4/sports/{sport_key}/scores", daysFrom=days_from, dateFormat="iso")
    return data if isinstance(data, list) else []

# ========================= Utils =========================
def fmt_time(iso):
    try: t = datetime.fromisoformat((iso or "").replace("Z","+00:00")).astimezone(); return t.strftime("%b %d, %I:%M %p")
    except Exception: return iso or "TBD"
def parse_home_away(ev):
    home = ev.get("home_team","TBD")
    away = ev.get("away_team")
    if not away:
        teams = ev.get("teams",[])
        away = next((t for t in teams if t!=home), teams[0] if teams else "TBD")
    return home, away
def label_for_event(ev):
    home, away = parse_home_away(ev); return f"{away} @ {home} — {fmt_time(ev.get('commence_time','TBD'))}"
def best_ml_prices(ev, only_book=None):
    best = {}
    for bk in ev.get("bookmakers", []):
        title = (bk.get("title") or "").strip()
        if only_book and title.lower() != str(only_book).lower(): continue
        for m in bk.get("markets", []):
            if m.get("key") != "h2h": continue
            for o in m.get("outcomes", []):
                team, price = o.get("name"), o.get("price")
                if team is None or price is None: continue
                if team not in best or price > best[team]["price"]:
                    best[team] = {"price": price, "bookmaker": title or only_book or "Best"}
    return best
def implied_from_decimal(dec_a, dec_b):
    try:
        a=1.0/float(dec_a); b=1.0/float(dec_b); s=a+b
        return (a/s, b/s) if s>0 else (None,None)
    except Exception: return (None,None)
def conf_meter(p):
    p=float(p or 0); blocks=max(0,min(10,int(round(p*10)))); return "■"*blocks + "□"*(10-blocks)

# ========================= Model =========================
def compute_last_n(team, scores, n=10):
    games=[g for g in scores if g.get("completed") and team in (g.get("teams") or [g.get("home_team"),g.get("away_team")])]
    games.sort(key=lambda g:g.get("commence_time") or "", reverse=True); recent=games[:n]
    wins=0; pf=0.0; pa=0.0
    for g in recent:
        h,a=g.get("home_team"),g.get("away_team"); sc=g.get("scores") or []
        sdict={x.get("name"):float(x.get("score")) for x in sc if x.get("name") is not None and x.get("score") is not None}
        winner = h if sdict.get(h,0)>sdict.get(a,0) else a
        if winner==team: wins+=1
        pf+=sdict.get(team,0.0); opp=a if team==h else h; pa+=sdict.get(opp,0.0)
    gp=len(recent) or 1
    return wins, len(recent), (pf-pa)/gp

def compute_h2h(a,b,scores,n=6):
    games=[g for g in scores if g.get("completed") and a in (g.get("teams") or [g.get("home_team"),g.get("away_team")]) and b in (g.get("teams") or [g.get("home_team"),g.get("away_team")])]
    games.sort(key=lambda g:g.get("commence_time") or "", reverse=True); games=games[:n]
    aw=0
    for g in games:
        h=g.get("home_team"); a2=g.get("away_team"); sc=g.get("scores") or []
        sdict={x.get("name"):float(x.get("score")) for x in sc if x.get("name") is not None and x.get("score") is not None}
        winner = h if sdict.get(h,0)>sdict.get(a2,0) else a2
        if winner==a: aw+=1
    return aw, len(games)

def compute_elo(scores, k=20.0, hfa=30.0):
    teams={}
    ordered=sorted([g for g in scores if g.get("completed")], key=lambda g:g.get("commence_time") or "")
    for g in ordered:
        h,a=g.get("home_team"),g.get("away_team")
        if not h or not a: continue
        teams.setdefault(h,1500.0); teams.setdefault(a,1500.0)
        sc=g.get("scores") or []
        sdict={x.get("name"):float(x.get("score")) for x in sc if x.get("name") is not None and x.get("score") is not None}
        hs=sdict.get(h,0.0); as_=sdict.get(a,0.0)
        Rh,Ra=teams[h],teams[a]
        Eh=1.0/(1.0+10**((Ra-(Rh+hfa))/400.0)); Ea=1.0-Eh
        Sh,Sa=(1.0,0.0) if hs>as_ else ((0.0,1.0) if hs<as_ else (0.5,0.5))
        teams[h]=Rh+k*(Sh-Eh); teams[a]=Ra+k*(Sa-Ea)
    return teams

def rest_info(team, scores, as_of_iso):
    try: as_of=datetime.fromisoformat(as_of_iso.replace("Z","+00:00"))
    except Exception: as_of=datetime.utcnow()
    games=[g for g in scores if g.get("completed") and team in (g.get("teams") or [g.get("home_team"),g.get("away_team")])]
    games.sort(key=lambda g:g.get("commence_time") or "", reverse=True)
    if not games: return 7, False
    last=games[0]
    try: last_time=datetime.fromisoformat(last.get("commence_time","").replace("Z","+00:00"))
    except Exception: last_time=as_of-timedelta(days=2)
    days=max(0,(as_of-last_time).days); b2b=(as_of-last_time)<=timedelta(hours=36)
    return days, b2b

def injuries_and_lineups(group, team_name, date_iso, sport_key):
    if group == "Soccer":
        return soccer_injuries_and_lineups(team_name, date_iso, sport_key)
    if sport_key in SDIO_SPORT_FOR_KEY and SPORTSDATA_API_KEY:
        return sdio_injuries_for_team(sport_key, team_name), []
    return [], []

def implied_from_decimal(dec_a, dec_b):
    try:
        a=1.0/float(dec_a); b=1.0/float(dec_b); s=a+b
        return (a/s, b/s) if s>0 else (None,None)
    except Exception: return (None,None)

def best_ml_prices(ev, only_book=None):
    best = {}
    for bk in ev.get("bookmakers", []):
        title = (bk.get("title") or "").strip()
        if only_book and title.lower() != str(only_book).lower(): continue
        for m in bk.get("markets", []):
            if m.get("key") != "h2h": continue
            for o in m.get("outcomes", []):
                team, price = o.get("name"), o.get("price")
                if team is None or price is None: continue
                if team not in best or price > best[team]["price"]:
                    best[team] = {"price": price, "bookmaker": title or only_book or "Best"}
    return best

def conf_meter(p):
    p=float(p or 0); blocks=max(0,min(10,int(round(p*10)))); return "■"*blocks + "□"*(10-blocks)

def edgefinder_predict(home_team, away_team, sport_key, event, scores_cache=None, elo_table=None, only_book=None):
    group = group_from_key(sport_key)
    when_iso = event.get("commence_time") or ""

    best = best_ml_prices(event, only_book=only_book)
    dec_home = best.get(home_team, {}).get("price"); dec_away = best.get(away_team, {}).get("price")
    mh, ma = implied_from_decimal(dec_home, dec_away) if (dec_home is not None and dec_away is not None) else (None,None)

    eh = gap = fh = fa = hh = None
    rest_h = rest_a = 3; b2b_h=b2b_a=False
    if scores_cache:
        if elo_table is None: elo_table = compute_elo(scores_cache)
        Rh = elo_table.get(home_team,1500.0); Ra=elo_table.get(away_team,1500.0)
        eh = 1.0/(1.0+10**((Ra-(Rh+30.0))/400.0)); gap=Rh-Ra
        w_h,n_h,_ = compute_last_n(home_team,scores_cache,10)
        w_a,n_a,_ = compute_last_n(away_team,scores_cache,10)
        fh = (w_h/n_h) if n_h else None; fa=(w_a/n_a) if n_a else None
        a_wins,n_h2h = compute_h2h(away_team,home_team,scores_cache,6)
        hh = (1.0 - a_wins/n_h2h) if n_h2h else None
        rest_h,b2b_h = rest_info(home_team,scores_cache,when_iso)
        rest_a,b2b_a = rest_info(away_team,scores_cache,when_iso)

    inj_home, lineup_home = injuries_and_lineups(group, home_team, when_iso, sport_key)
    inj_away, lineup_away = injuries_and_lineups(group, away_team, when_iso, sport_key)

    weights=[]; probs_home=[]
    if mh is not None: weights.append(0.45); probs_home.append(mh)
    if eh is not None: weights.append(0.25); probs_home.append(eh)
    if fh is not None and fa is not None:
        form_prob=0.5+(fh-fa)*0.12; weights.append(0.15); probs_home.append(form_prob)
    if hh is not None: weights.append(0.05); probs_home.append(hh)

    inj_adj=0.0
    if inj_home or inj_away:
        if len(inj_home)>=2 and len(inj_away)<2: inj_adj-=0.02
        if len(inj_away)>=2 and len(inj_home)<2: inj_adj+=0.02
    rest_bump=0.0
    if rest_h is not None and rest_a is not None:
        rest_bump+=0.02*(rest_h-rest_a)
    if b2b_a and not b2b_h: rest_bump+=0.03
    if b2b_h and not b2b_a: rest_bump-=0.03
    if weights:
        ph=sum(p*w for p,w in zip(probs_home,weights))/sum(weights)
        ph=max(0.0,min(1.0,ph+max(-0.05,min(0.05,rest_bump+inj_adj))))
    else:
        ph=0.55

    pick_team = home_team if ph>=0.5 else away_team
    conf = ph if pick_team==home_team else (1.0-ph)

    reasons=[]
    if mh is not None:
        fav_price = dec_home if pick_team==home_team else dec_away
        dog_price = dec_away if pick_team==home_team else dec_home
        reasons.append(f"Market edge: fav {fav_price} vs dog {dog_price}")
    if eh is not None: reasons.append(f"ELO gap {gap:+.0f} (favours {'home' if gap>=0 else 'away'})")
    if fh is not None and fa is not None:
        reasons.append(f"Form last 10: {home_team} {int((fh or 0)*10)}-{10-int((fh or 0)*10)} | {away_team} {int((fa or 0)*10)}-{10-int((fa or 0)*10)}")
    if hh is not None: reasons.append(f"Head-to-head (recent) home share {hh:.0%}")
    if inj_home: reasons.append(f"Injuries {home_team}: " + ", ".join(inj_home[:5]) + ("..." if len(inj_home)>5 else ""))
    if inj_away: reasons.append(f"Injuries {away_team}: " + ", ".join(inj_away[:5]) + ("..." if len(inj_away)>5 else ""))
    if lineup_home: reasons.append(f"Probable starters {home_team}: " + ", ".join(lineup_home[:5]) + ("..." if len(lineup_home)>5 else ""))
    if lineup_away: reasons.append(f"Probable starters {away_team}: " + ", ".join(lineup_away[:5]) + ("..." if len(lineup_away)>5 else ""))
    reasons.append("Home advantage favors " + (home_team if pick_team==home_team else pick_team) if pick_team==home_team else f"{pick_team} favored despite travel")
    return {"pick_team": pick_team, "conf": float(conf), "reasons": reasons}

# ========================= UI =========================
st.title("EdgeFinder — Pro")

regions_map={"US":"us","UK":"uk","EU":"eu","AU":"au"}
regions_sel=st.multiselect("Regions",["US","UK","EU","AU"],default=["US","EU"])
regions_param=",".join(regions_map[x] for x in regions_sel) or "us"

col1,col2=st.columns([1,1])
with col1: mode_by_sport=st.toggle("Run by sport (aggregate all leagues)",value=True)
with col2: only_dk=st.toggle("DraftKings only",value=False)
only_book="DraftKings" if only_dk else None

if mode_by_sport:
    key_to_group, SPORT_GROUPS, LEAGUES = sports_index()
    group = st.selectbox("Sport", SPORT_GROUPS, index=SPORT_GROUPS.index(DEFAULT_GROUP) if DEFAULT_GROUP in SPORT_GROUPS else 0)
else:
    league_names=[name for _,name in LEAGUES] or ["(No leagues loaded)"]
    league=st.selectbox("League", league_names)
    name_to_key={name:key for key,name in LEAGUES}

if st.button("Refresh games", use_container_width=True):
    st.session_state.pop("events", None)

if "events" not in st.session_state:
    if mode_by_sport:
        sport_keys=[k for k,g in key_to_group.items() if g==group]
        merged=[]
        for sk in sport_keys:
            try: merged.extend(fetch_league_robust(sk, regions=regions_param) or [])
            except Exception: pass
        st.session_state["events"]=merged
    else:
        key = name_to_key.get(league, None) if not mode_by_sport else None
        st.session_state["events"]=fetch_league_robust(key, regions=regions_param) if key else []

events=st.session_state.get("events", [])

scores_cache=[]; elo_table={}
if events:
    sk_first=events[0].get("sport_key","")
    scores_cache=fetch_scores(sk_first, days_from=120)
    elo_table=compute_elo(scores_cache)

if not events:
    st.info("No upcoming events for the current filters.")
else:
    q=st.text_input("Filter by team (optional)")
    filtered=[e for e in events if not q or q.lower() in label_for_event(e).lower()]
    if not filtered:
        st.info("No events match the filter.")
    else:
        labels=[label_for_event(e) for e in filtered]
        ids=list(range(len(filtered)))
        c1,c2=st.columns([1,1])
        with c1: select_all=st.checkbox("Select all", value=(len(filtered)<=10))
        selected=ids if select_all else st.multiselect("Select matchup(s)", ids, format_func=lambda i: labels[i])
        if selected and st.button("Run Predictions", type="primary", use_container_width=True):
            rows=[]
            for i in selected:
                ev=filtered[i]; home,away=parse_home_away(ev)
                pred=edgefinder_predict(home,away,ev.get("sport_key",""),ev,scores_cache=scores_cache,elo_table=elo_table,only_book=only_book)
                pick_team=pred.get("pick_team",home); conf=float(pred.get("conf",0.0)); reasons=list(pred.get("reasons",[]))
                best=best_ml_prices(ev, only_book=only_book); dec_home=best.get(home,{}).get("price"); dec_away=best.get(away,{}).get("price")
                when_txt=fmt_time(ev.get("commence_time")); odds_line=f"{home} {dec_home if dec_home is not None else '—'} | {away} {dec_away if dec_away is not None else '—'}"
                ball = "🏀" if ev.get("sport_key","").startswith("basketball_") else ("⚽️" if ev.get("sport_key","").startswith("soccer_") else "🎯")
                st.markdown(
                    f"{ball} **{away} vs {home}**\n"
                    f"───────────────────────────────────────\n"
                    f"📅 {when_txt}\n"
                    f"📈 **Odds (best price):** {odds_line}\n"
                    f"🤖 **Prediction:** **{pick_team}** favored to win ({(conf or 0):.0%})\n"
                    f"🟢 **Confidence Meter:** {conf_meter(conf or 0)}\n\n"
                    f"**💬 Detailed Reasoning:**"
                )
                for r in reasons[:12]: st.markdown("• " + r)
                verdict = "Strong" if (conf or 0)>=0.65 else ("Medium" if (conf or 0)>=0.55 else "Lean")
                st.markdown(f"───────────────────────────────────────\n✅ **EdgeFinder Model Verdict:** {verdict} edge for **{pick_team}**")
                rows.append({"When":when_txt,"Sport":ev.get("sport_key"),"Matchup":f"{away} @ {home}","Model Pick":pick_team,"Shown Prob.":f"{(conf or 0):.0%}","Best Home ML":dec_home,"Best Away ML":dec_away})
            if rows:
                df=pd.DataFrame(rows); csv=df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV (summary)", data=csv, file_name="edgefinder_cards.csv", mime="text/csv")
