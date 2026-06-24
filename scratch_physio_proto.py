import pandas as pd, numpy as np
D="data/ch2025_data_items/"

def night_assign(df):
    h=df["timestamp"].dt.hour; d=df["timestamp"].dt.normalize()
    nd=pd.Series(pd.NaT,index=df.index,dtype="datetime64[ns]")
    nd[h>=18]=d[h>=18]+pd.Timedelta(days=1); nd[h<12]=d[h<12]
    df=df[nd.notna()].copy(); df["night"]=nd[nd.notna()]
    hh=df["timestamp"].dt.hour+df["timestamp"].dt.minute/60.0
    df["axis"]=hh.where(hh<18,hh-24)  # -6 .. +12
    df["minute"]=(df["axis"]*60).round().astype(int)  # -360 .. +720
    return df

# Load minute-level streams, restrict to a few subjects for prototype
subs=["id01","id02","id06","id05"]  # id06 high S, id05 low S
def load(name,cols):
    df=pd.read_parquet(D+f"ch2025_{name}.parquet",columns=["subject_id","timestamp",*cols])
    df=df[df.subject_id.isin(subs)]
    return night_assign(df)

scr=load("mScreenStatus",["m_screen_use"])
act=load("mActivity",["m_activity"])
pedo=load("wPedo",["step"])
hr=load("wHr",["heart_rate"])
arr=hr["heart_rate"].apply(lambda a:np.asarray(a,float))
hr["hr_mean"]=arr.apply(lambda a:a.mean() if a.size else np.nan)

# build minute panel per (subject,night)
def panel(subject,night):
    idx=np.arange(-360,720)  # 18:00..12:00
    p=pd.DataFrame(index=idx)
    def grab(df,col,agg="max"):
        s=df[(df.subject_id==subject)&(df.night==night)]
        if s.empty: return pd.Series(np.nan,index=idx)
        g=s.groupby("minute")[col].agg(agg)
        return g.reindex(idx)
    p["screen"]=grab(scr,"m_screen_use","max")
    p["still"]=(grab(act,"m_activity","min").isin([3])).astype(float) # crude
    a=act[(act.subject_id==subject)&(act.night==night)]
    if not a.empty:
        gm=a.groupby("minute")["m_activity"]
        p["move"]=gm.apply(lambda s:s.isin([1,2,7,8]).any()).reindex(idx).astype(float)
        p["still"]=gm.apply(lambda s:(s==3).mean()).reindex(idx)
    p["step"]=grab(pedo,"step","sum")
    p["hr"]=grab(hr,"hr_mean","mean")
    return p

def estimate(subject,night):
    p=panel(subject,night)
    # resting HR = 10th pct of HR over night
    rest=np.nanpercentile(p["hr"],10) if p["hr"].notna().sum()>20 else np.nan
    # awake score per minute
    awake = ((p["step"].fillna(0)>0) | (p["move"].fillna(0)>0) | (p["screen"].fillna(0)>0)
             | ((p["hr"]> (rest+8)) if np.isfinite(rest) else False)).astype(float)
    # asleep candidate: not awake AND has some evidence of wear/quiet (hr present or still)
    quiet=(1-awake)
    # main sleep block: longest run of quiet allowing gaps -> smooth with rolling
    sm=quiet.rolling(11,center=True,min_periods=1).mean()
    asleep=(sm>=0.5).astype(int).values
    # find longest contiguous asleep run
    best=(0,0,0); i=0; n=len(asleep)
    while i<n:
        if asleep[i]==1:
            j=i
            while j<n and asleep[j]==1: j+=1
            if j-i>best[0]: best=(j-i,i,j)
            i=j
        else: i+=1
    length,s,e=best
    idx=p.index.values
    if length<60: # <1h no valid sleep
        return dict(subject=subject,night=str(night.date()),tst=np.nan)
    onset=idx[s]; wake=idx[e-1]
    block=p.iloc[s:e]
    waso=int((awake.iloc[s:e]>0).sum())
    tst=(length-waso)/60.0  # hours
    tib=(wake-onset)/60.0
    se=tst/tib if tib>0 else np.nan
    # SOL: from first quiet after last big activity before onset. proxy: minutes from "lights/screen off settle" 
    # use: last screen-on before onset -> onset
    pre=p.iloc[:s]
    last_screen=pre.index[pre["screen"].fillna(0)>0]
    bedtime= last_screen.max() if len(last_screen) else onset
    sol=max(0,(onset-bedtime))
    n_awak=int(((awake.iloc[s:e].values[1:]==1)&(awake.iloc[s:e].values[:-1]==0)).sum())
    return dict(subject=subject,night=str(night.date()),onset_h=onset/60,wake_h=wake/60,
                tst=round(tst,2),se=round(se,3),sol=int(sol),waso=waso,n_awak=n_awak,rest=round(rest,1) if np.isfinite(rest) else np.nan)

# run on labeled nights and compare to labels
train=pd.read_csv("data/ch2026_metrics_train.csv",parse_dates=["sleep_date"])
rows=[]
for sub in subs:
    nights=sorted(train[train.subject_id==sub]["sleep_date"].unique())[:25]
    for nt in nights:
        nt=pd.Timestamp(nt)
        rows.append(estimate(sub,nt))
est=pd.DataFrame(rows).dropna(subset=["tst"])
print("=== TST distribution (hours) ===")
print(est["tst"].describe().round(2).to_string())
print("\n=== sample estimates ===")
print(est.head(20).to_string())
m=est.merge(train,left_on=["subject","night"],right_on=[train["subject_id"],train["sleep_date"].astype(str).str[:10]],how="left")
