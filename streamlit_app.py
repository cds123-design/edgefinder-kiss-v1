
import streamlit as st, pandas as pd, numpy as np, requests

st.set_page_config(page_title="EdgeFinder KISS v1 — Moneyline Predictor", layout="wide")
ALLOWED_BOOKS=["draftkings","fanduel","espnbet"]; SPORT_KEY_DEFAULT="basketball_nba"

def implied_prob(d):
    try:
        d=float(d); 
        return 1.0/d if d>1 else np.nan
    except: 
        return np.nan

def normalize_probs(pa,pb):
    s=pa+pb
    if s<=0 or np.isnan(s): return np.nan,np.nan
    return pa/s,pb/s

def best_prices_from_events(events_json,allowed_books=ALLOWED_BOOKS):
    best={}
    for ev in events_json:
        h,a=ev.get("home_team"),ev.get("away_team")
        if not h or not a: 
            continue
        bh=(None,None); ba=(None,None)
        for bk in ev.get("bookmakers",[]):
            if allowed_books and bk.get("key") not in allowed_books: 
                continue
            for m in bk.get("markets",[]):
                if m.get("key")!="h2h": continue
                for o in m.get("outcomes",[]):
                    name,price=o.get("name"),o.get("price")
                    if not price: continue
                    if name==h and (bh[0] is None or price>bh[0]): bh=(price,bk.get("key"))
                    if name==a and (ba[0] is None or price>ba[0]): ba=(price,bk.get("key"))
        if bh[0] and ba[0]:
            best[(h,a)]={"home_price":bh[0],"away_price":ba[0],"home_book":bh[1],"away_book":ba[1]}
    return best

@st.cache_data(ttl=900)
def fetch_odds(sport_key,region="us",market="h2h"):
    api_key=st.secrets.get("ODDS_API_KEY","")
    if not api_key: 
        raise RuntimeError("Missing ODDS_API_KEY in Secrets.")
    url=f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    r=requests.get(url,params={"regions":region,"markets":market,"oddsFormat":"decimal","apiKey":api_key},timeout=20)
    r.raise_for_status()
    return r.json()

def match_event_best_price(best,home,away):
    hq,aq=home.lower().strip(),away.lower().strip()
    for (H,A),v in best.items():
        if hq in H.lower() and aq in A.lower(): return H,A,v
        if hq in A.lower() and aq in H.lower(): return A,H,{"home_price":v["away_price"],"away_price":v["home_price"],"home_book":v["away_book"],"away_book":v["home_book"]}
    return None

def meter(p):
    if np.isnan(p): return "⚪ Unknown"
    if p>=0.66: return "🟢 High"
    if p>=0.55: return "🟡 Medium"
    return "🟠 Lean"

def reasons(home,away,ph,pa,is_home=True,notes=None):
    R=[]
    if not np.isnan(ph) and not np.isnan(pa):
        R.append(f"The market leans toward {home if ph>pa else away} at the current price.")
    if is_home: R.append(f"{home} get home-court advantage.")
    if notes: 
        n=str(notes).strip()
        if n: R.append(n)
    if not np.isnan(ph) and not np.isnan(pa):
        d=abs(ph-pa)
        if d>=0.10: R.append("Pricing gap suggests a clear favorite tonight.")
        elif d>=0.05: R.append("Odds indicate a modest but real edge.")
        else: R.append("Books price this fairly close; expect volatility.")
    return R[:4]

def model_prob(ph,pa):
    if np.isnan(ph) or np.isnan(pa): return np.nan,np.nan
    return normalize_probs(ph,pa)

tabs=st.tabs(["Predict (Single)","Bulk Upload","Backtest","About"])

with tabs[0]:
    st.header("Predict (Single)")
    c1,c2=st.columns(2)
    home=c1.text_input("Home team","Los Angeles Clippers")
    away=c2.text_input("Away team","Sacramento Kings")
    sport=st.text_input("Sport key (The Odds API)",SPORT_KEY_DEFAULT)
    if st.button("Run Prediction"):
        try:
            events=fetch_odds(sport); best=best_prices_from_events(events,ALLOWED_BOOKS)
            m=match_event_best_price(best,home,away)
            if not m: st.error("Couldn't find odds for that matchup. Check team names or sport key.")
            else:
                H,A,bp=m; ho,ao=bp["home_price"],bp["away_price"]
                ph,pa=implied_prob(ho),implied_prob(ao); phn,pan=model_prob(ph,pa)
                win=H if phn>=pan else A; conf=max(phn,pan)
                st.subheader(f"{H} vs {A}")
                st.write(f"**Odds (best price):** {H} {ho:.2f}  |  {A} {ao:.2f}")
                st.write(f"**Model Prediction:** **{win}** favored to win ({conf:.0%})")
                st.write(f"**Confidence:** {meter(conf)}")
                st.markdown("**Reasoning:**")
                for r in reasons(H,A,phn,pan,True): st.write(f"- {r}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

with tabs[1]:
    st.header("Bulk Upload")
    st.write("Upload CSV: home_team,away_team (optional notes)")
    demo="home_team,away_team,notes\nHouston Rockets,Atlanta Hawks,Hawks resting starters\nDallas Mavericks,Los Angeles Lakers,Back-to-back for Lakers"
    st.download_button("Download CSV template",data=demo,file_name="kiss_template.csv",mime="text/csv")
    up=st.file_uploader("Upload CSV",type=["csv"])
    if up is not None:
        try:
            df=pd.read_csv(up)
            if not {"home_team","away_team"}.issubset(df.columns): st.error("Missing columns: home_team, away_team")
            else:
                events=fetch_odds(SPORT_KEY_DEFAULT); best=best_prices_from_events(events,ALLOWED_BOOKS)
                rows=[]
                for _,r in df.iterrows():
                    H,A=str(r["home_team"]),str(r["away_team"]); notes=str(r.get("notes","")).strip()
                    m=match_event_best_price(best,H,A)
                    if not m: rows.append({"matchup":f"{H} vs {A}","odds":"—","prediction":"No data","confidence":"","reason":"Odds not found"}); continue
                    h,a,bp=m; ho,ao=bp["home_price"],bp["away_price"]; ph,pa=implied_prob(ho),implied_prob(ao); phn,pan=model_prob(ph,pa)
                    win=h if phn>=pan else a; conf=max(phn,pan); rs=" | ".join(reasons(h,a,phn,pan,True,notes))
                    rows.append({"matchup":f"{h} vs {a}","odds":f"{h} {ho:.2f} / {a} {ao:.2f}","prediction":win,"confidence":f"{conf:.0%}","reason":rs})
                st.dataframe(pd.DataFrame(rows),use_container_width=True)
        except Exception as e:
            st.error(f"Bulk prediction error: {e}")

with tabs[2]:
    st.header("Backtest")
    st.write("Upload: home_team,away_team,home_odds,away_odds,actual_winner")
    bt=st.file_uploader("Upload backtest CSV",type=["csv"],key="bt")
    if bt is not None:
        try:
            df=pd.read_csv(bt)
            if not {"home_team","away_team","home_odds","away_odds","actual_winner"}.issubset(df.columns): st.error("Missing columns for backtest.")
            else:
                hits=[]
                for _,r in df.iterrows():
                    ph,pa=implied_prob(r["home_odds"]),implied_prob(r["away_odds"]); phn,pan=model_prob(ph,pa)
                    win=r["home_team"] if phn>=pan else r["away_team"]
                    hits.append(int(str(win).lower()==str(r["actual_winner"]).lower()))
                st.write(f"**Model Accuracy:** {np.mean(hits):.0%}" if hits else "No rows parsed.")
        except Exception as e:
            st.error(f"Backtest error: {e}")

with tabs[3]:
    st.header("About")
    st.write("EdgeFinder KISS v1 — clean moneyline predictor with live odds, natural-language reasoning, and optional backtest.")
