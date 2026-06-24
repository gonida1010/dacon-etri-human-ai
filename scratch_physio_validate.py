import pandas as pd, numpy as np
D="data/ch2025_data_items/"
train=pd.read_csv("data/ch2026_metrics_train.csv",parse_dates=["sleep_date"])
subs=list(train.subject_id.unique())

def night_assign(df):
    h=df["timestamp"].dt.hour; d=df["timestamp"].dt.normalize()
    nd=pd.Series(pd.NaT,index=df.index,dtype="datetime64[ns]")
    nd[h>=18]=d[h>=18]+pd.Timedelta(days=1); nd[h<12]=d[h<12]
    df=df[nd.notna()].copy(); df["night"]=nd[nd.notna()]
    hh=df["timestamp"].dt.hour+df["timestamp"].dt.minute/60.0
    df["minute"]=((hh.where(hh<18,hh-24))*60).round().astype(int)
    return df
def load(name,cols):
    df=pd.read_parquet(D+f"ch2025_{name}.parquet",columns=["subject_id","timestamp",*cols])
    return night_assign(df)
print("loading...",flush=True)
scr=load("mScreenStatus",["m_screen_use"]); scr=scr.groupby(["subject_id","night","minute"])["m_screen_use"].max()
act=load("mActivity",["m_activity"])
act_move=act.assign(move=act.m_activity.isin([1,2,7,8]).astype(float)).groupby(["subject_id","night","minute"])["move"].max()
pedo=load("wPedo",["step"]); pedo=pedo.groupby(["subject_id","night","minute"])["step"].sum()
hr=load("wHr",["heart_rate"]); 
hr["hrm"]=hr["heart_rate"].apply(lambda a: float(np.mean(a)) if len(a) else np.nan)
hrm=hr.groupby(["subject_id","night","minute"])["hrm"].mean()
print("panels built",flush=True)

idxg=np.arange(-360,720)
def estimate(subject,night):
    def g(s): 
        try: v=s.loc[(subject,night)]; return v.reindex(idxg)
        except KeyError: return pd.Series(np.nan,index=idxg)
    screen=g(scr).fillna(0); move=g(act_move).fillna(0); step=g(pedo).fillna(0); h=g(hrm)
    if h.notna().sum()<30: return None
    rest=np.nanpercentile(h,10)
    awake=((step>0)|(move>0)|(screen>0)|(h>rest+8)).astype(float)
    quiet=1-awake
    sm=quiet.rolling(15,center=True,min_periods=1).mean()
    asleep=(sm>=0.5).astype(int).values
    # longest run
    best=(0,0,0);i=0;n=len(asleep)
    while i<n:
        if asleep[i]:
            j=i
            while j<n and asleep[j]:j+=1
            if j-i>best[0]:best=(j-i,i,j)
            i=j
        else:i+=1
    L,s,e=best
    if L<90: return None
    onset=idxg[s]; wake=idxg[e-1]
    waso=int((awake.values[s:e]>0).sum())
    tst=(L-waso)/60.0
    # fragmentation: awake transitions in window
    aw=awake.values[s:e]; nawak=int(((aw[1:]==1)&(aw[:-1]==0)).sum())
    se=(L-waso)/L
    # SOL proxy: from first quiet settle (after last evening screen burst) to onset
    pre_screen=np.where(screen.values[:s]>0)[0]
    bed = idxg[pre_screen.max()] if len(pre_screen) else onset
    sol=max(0,onset-bed)
    se2=tst/((wake-bed)/60.0) if wake-bed>0 else np.nan
    # HR depth: how low HR drops vs awake baseline
    hr_sleep=np.nanmean(h.values[s:e]); hr_evening=np.nanmean(h.values[max(0,s-120):s])
    return dict(subject=subject,sleep_date=str(pd.Timestamp(night).date()),
                tst=tst,se=se,se2=se2,sol=sol,waso=waso,nawak=nawak,
                onset_h=onset/60,wake_h=wake/60,rest=rest,
                hr_sleep=hr_sleep,hr_drop=(hr_evening-hr_sleep) if np.isfinite(hr_evening) else np.nan,
                frag=nawak/(L/60.0))

rows=[]
for sub in subs:
    nights=sorted(train[train.subject_id==sub]["sleep_date"].unique())
    for nt in nights:
        r=estimate(sub,pd.Timestamp(nt))
        if r: rows.append(r)
est=pd.DataFrame(rows)
print("estimated nights:",len(est),flush=True)
tr=train.copy(); tr["sleep_date"]=tr["sleep_date"].astype(str).str[:10]
m=est.merge(tr,on=["subject_id" if False else "subject","sleep_date"],how="inner",
            left_on=["subject","sleep_date"],right_on=["subject_id","sleep_date"]) if False else \
  est.rename(columns={"subject":"subject_id"}).merge(tr,on=["subject_id","sleep_date"],how="inner")
print("merged:",len(m))
print("\n=== Point-biserial corr: physio estimate vs S labels ===")
feats=["tst","se","se2","sol","waso","nawak","frag","onset_h","wake_h","rest","hr_sleep","hr_drop"]
for t in ["S1","S2","S3","S4","Q1","Q2","Q3"]:
    cs=[]
    for f in feats:
        v=m[[f,t]].dropna()
        if len(v)>30 and v[f].std()>0:
            c=np.corrcoef(v[f],v[t])[0,1]; cs.append((f,c))
    cs.sort(key=lambda x:-abs(x[1]))
    print(f"{t}: "+", ".join(f"{f}={c:+.2f}" for f,c in cs[:4]))
print("\n=== NSF threshold separation for S1 (TST 7-9h) ===")
m["tst_ok"]=((m.tst>=6.5)&(m.tst<=9)).astype(int)
print(m.groupby("tst_ok")["S1"].mean())
print("\n=== S2 by SE>=0.85 ===")
m["se_ok"]=(m.se2>=0.85).astype(int)
print(m.groupby("se_ok")["S2"].agg(["mean","count"]))
