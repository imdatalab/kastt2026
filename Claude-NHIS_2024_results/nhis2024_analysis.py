# -*- coding: utf-8 -*-
"""
국민건강보험공단(NHIS) 2024년 건강검진정보 데이터 분석
=======================================================
Author  : Statistical Consulting (for Hwamin)
Purpose : 기술통계, 성별 비교, 연령대별 BMI 분포, 대사증후군 구성요소 상관분석,
          흡연상태별 간기능지표 비교
Environment: Google Colab (Python 3.10+)
Reproducibility: random_state = 42 전역 고정

패키지 버전 (분석 시점 기준, 재현성을 위해 명시)
  pandas 2.x / numpy 1.26.x / scipy 1.17.x / seaborn 0.13.x / matplotlib 3.10.x

사용 방법 (Colab):
  1) 좌측 파일 탭에서 '2_NHIS_2024.csv' 업로드 (또는 Google Drive 마운트 후 경로 수정)
  2) 아래 CONFIG.data_path 를 실제 업로드 경로로 수정
  3) 전체 셀 실행 (Runtime > Run all)
"""

import os
import warnings
from itertools import combinations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class Config:
    data_path: str = "2_NHIS_2024.csv"          # Colab 업로드 후 경로
    encoding: str = "cp949"                       # 공공데이터포털 CSV는 대개 EUC-KR/CP949
    output_dir: str = "./outputs"
    dpi: int = 300


CFG = Config()
os.makedirs(CFG.output_dir, exist_ok=True)

# Okabe-Ito 컬러블라인드 친화 팔레트
OKABE_ITO = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7"}


# ----------------------------------------------------------------------
# 0. 그래픽 환경 설정 (한글 폰트 + 의학저널 스타일)
# ----------------------------------------------------------------------
def setup_plot_style():
    """
    Purpose: Colab/Linux 환경에서 한글이 깨지지 않도록 폰트를 등록하고,
             NEJM/Lancet 계열 논문에서 흔히 쓰는 미니멀 스타일을 전역 적용한다.
    Input  : None
    Output : None (matplotlib rcParams 갱신)
    """
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Colab/Ubuntu 기본 Noto CJK
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",         # 나눔고딕 (있는 경우)
    ]
    font_name = None
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            break

    if font_name is None:
        # Colab에서 나눔고딕이 없을 경우 자동 설치 (최초 1회, 인터넷 연결 필요)
        os.system("apt-get -qq install -y fonts-nanum > /dev/null 2>&1")
        os.system("fc-cache -fv > /dev/null 2>&1")
        path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()

    if font_name:
        plt.rcParams["font.family"] = font_name
    else:
        warnings.warn("한글 폰트를 찾지 못했습니다. 라벨이 깨질 수 있습니다.")

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "savefig.dpi": CFG.dpi,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    })


# ----------------------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ----------------------------------------------------------------------
COLMAP = {
    "기준년도": "year", "가입자일련번호": "id", "시도코드": "region_code", "성별코드": "sex",
    "연령대코드(5세단위)": "age_group", "신장(5cm단위)": "height", "체중(5kg단위)": "weight",
    "허리둘레": "waist", "시력(좌)": "vision_l", "시력(우)": "vision_r",
    "청력(좌)": "hearing_l", "청력(우)": "hearing_r", "수축기혈압": "sbp", "이완기혈압": "dbp",
    "식전혈당(공복혈당)": "glucose", "총콜레스테롤": "tchol", "트리글리세라이드": "tg",
    "HDL콜레스테롤": "hdl", "LDL콜레스테롤": "ldl", "혈색소": "hgb", "요단백": "proteinuria",
    "혈청크레아티닌": "creatinine", "혈청지오티(AST)": "ast", "혈청지피티(ALT)": "alt",
    "감마지티피": "ggt", "흡연상태": "smoking", "음주여부": "drinking", "구강검진수검여부": "dental_exam",
    "치아우식증유무": "caries", "치석": "calculus",
}

AGE_LABELS = {5: "20-24", 6: "25-29", 7: "30-34", 8: "35-39", 9: "40-44", 10: "45-49",
              11: "50-54", 12: "55-59", 13: "60-64", 14: "65-69", 15: "70-74",
              16: "75-79", 17: "80-84", 18: "85+"}
AGE_ORDER = [AGE_LABELS[k] for k in sorted(AGE_LABELS)]

CONT_VARS = ["height", "weight", "bmi", "waist", "sbp", "dbp", "glucose", "tchol",
             "tg", "hdl", "ldl", "hgb", "creatinine", "ast", "alt", "ggt"]


def load_and_preprocess(path: str, encoding: str) -> pd.DataFrame:
    """
    Purpose: NHIS 원자료를 로드하고 분석용 파생변수를 생성한다.
    Input  : path(str) - CSV 경로, encoding(str) - 파일 인코딩 (기본 cp949)
    Output : pd.DataFrame - 전처리 완료 데이터
    """
    try:
        df = pd.read_csv(path, encoding=encoding)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"'{path}' 파일을 찾을 수 없습니다. Colab 좌측 파일 탭에서 업로드 후 "
            f"CFG.data_path를 실제 경로로 수정하세요."
        ) from e
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="euc-kr")

    df = df.rename(columns=COLMAP)

    # 데이터 검증: 100% 결측 컬럼 제거 (조사연도별 미포함 항목 - 데이터 설명서 참조)
    fully_missing = [c for c in df.columns if df[c].isnull().mean() == 1.0]
    if fully_missing:
        df = df.drop(columns=fully_missing)

    # 파생변수
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["age_group_label"] = df["age_group"].map(AGE_LABELS)
    df["sex_label"] = df["sex"].map({1: "Male", 2: "Female"})
    df["smoking_label"] = df["smoking"].map({1: "Never", 2: "Former", 3: "Current"})

    return df


def data_quality_report(df: pd.DataFrame) -> None:
    """
    Purpose: 결측률과 함께, 하위그룹 표본 불균형(예: 특정 연령대의 성별 쏠림) 등
             해석에 영향을 줄 수 있는 구조적 이슈를 사전에 점검한다.
    Input  : df - 전처리된 데이터프레임
    Output : None (콘솔 출력)
    """
    print("=" * 70)
    print("[데이터 품질 점검]")
    print("=" * 70)
    print(f"전체 표본 수: {len(df):,}")
    miss = (df[CONT_VARS].isnull().mean() * 100).round(2)
    print("\n연속형 변수 결측률(%):")
    print(miss[miss > 0] if (miss > 0).any() else " - 결측 없음(0%)")

    ct = pd.crosstab(df["age_group_label"], df["sex_label"]).reindex(AGE_ORDER)
    imbalanced = ct[(ct.min(axis=1) / ct.sum(axis=1)) < 0.05]
    if len(imbalanced) > 0:
        print("\n[경고] 다음 연령대는 성별 표본이 심하게 불균형합니다(소수 성별 비율 <5%).")
        print("       해당 구간의 성별 층화 결과는 해석에 주의가 필요합니다:")
        print(imbalanced)
    print()


# ----------------------------------------------------------------------
# 2. 기술통계
# ----------------------------------------------------------------------
def descriptive_statistics(df: pd.DataFrame, vars_: list) -> pd.DataFrame:
    """
    Purpose: 연속형 변수의 평균±SD, 중앙값[IQR], 결측률, 정규성 진단(왜도/첨도,
             Shapiro-Wilk 부분표본)을 산출한다.
    Input  : df - 데이터프레임, vars_ - 대상 변수 리스트
    Output : pd.DataFrame - 변수별 기술통계 요약표
    """
    rows = []
    for v in vars_:
        s = df[v].dropna()
        if len(s) == 0:
            continue
        sub = s.sample(min(5000, len(s)), random_state=RANDOM_STATE)  # scipy shapiro n<=5000 권장
        sh_stat, sh_p = stats.shapiro(sub)
        rows.append({
            "Variable": v, "N": len(s), "Missing(%)": round(df[v].isnull().mean() * 100, 2),
            "Mean": round(s.mean(), 2), "SD": round(s.std(), 2),
            "Median": round(s.median(), 2),
            "IQR": f"{s.quantile(.25):.2f}-{s.quantile(.75):.2f}",
            "Skewness": round(stats.skew(s), 2), "Kurtosis": round(stats.kurtosis(s), 2),
            "Shapiro_p(n=5000 subsample)": "<0.001" if sh_p < 0.001 else round(sh_p, 4),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. 성별 그룹비교 (정규성에 따라 t-test / Mann-Whitney U 자동 선택)
# ----------------------------------------------------------------------
def hedges_g(x: pd.Series, y: pd.Series):
    """Purpose: 두 독립표본 Cohen's d 및 표본크기 보정 Hedges' g 계산. Output: (d, g)"""
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / pooled_sd
    J = 1 - (3 / (4 * (nx + ny) - 9))
    return d, d * J


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Purpose: Mann-Whitney U 검정의 효과크기(rank-biserial correlation) 계산."""
    return 1 - (2 * u_stat) / (n1 * n2)


def compare_by_sex(df: pd.DataFrame, vars_: list, skew_threshold: float = 1.0) -> pd.DataFrame:
    """
    Purpose: 성별(Male/Female) 간 주요 지표를 비교한다. 표본 전체의 왜도가
             |skew|<1이면 정규근사 가능한 것으로 보아 Welch's t-test(등분산 가정 없음)를,
             그 외에는 이상치/편포에 강건한 Mann-Whitney U test를 적용한다.
             (참고: N>100,000 규모에서는 Shapiro 등 형식적 정규성 검정이 항상 유의하게
             나오므로, 왜도·첨도와 시각적 진단을 우선 기준으로 사용하는 것이 일반적 권고임)
    Input  : df, vars_, skew_threshold
    Output : pd.DataFrame - 검정방법/통계량/p-value/효과크기 요약표
    """
    male = df.loc[df["sex_label"] == "Male"]
    female = df.loc[df["sex_label"] == "Female"]
    rows = []
    for v in vars_:
        x, y = male[v].dropna(), female[v].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        skew_all = stats.skew(df[v].dropna())
        lev_stat, lev_p = stats.levene(x, y)

        if abs(skew_all) < skew_threshold:
            t_stat, p = stats.ttest_ind(x, y, equal_var=False)
            d, g = hedges_g(x, y)
            diff = x.mean() - y.mean()
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            rows.append({
                "Variable": v, "Test": "Welch's t-test", "N(M)": len(x), "N(F)": len(y),
                "Male": f"{x.mean():.2f}±{x.std():.2f}", "Female": f"{y.mean():.2f}±{y.std():.2f}",
                "Statistic": round(t_stat, 2), "Levene p": round(lev_p, 4),
                "p-value": "<0.001" if p < 0.001 else round(p, 3),
                "Effect size": f"g={g:.3f}",
                "95% CI (diff)": f"[{diff - 1.96*se:.2f}, {diff + 1.96*se:.2f}]",
            })
        else:
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            r = rank_biserial(u, len(x), len(y))
            rows.append({
                "Variable": v, "Test": "Mann-Whitney U", "N(M)": len(x), "N(F)": len(y),
                "Male": f"{x.median():.2f} [{x.quantile(.25):.2f}-{x.quantile(.75):.2f}]",
                "Female": f"{y.median():.2f} [{y.quantile(.25):.2f}-{y.quantile(.75):.2f}]",
                "Statistic": f"U={u:.3e}", "Levene p": round(lev_p, 4),
                "p-value": "<0.001" if p < 0.001 else round(p, 3),
                "Effect size": f"r={r:.3f}",
                "95% CI (diff)": "-",
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. 연령대별 BMI 분포 (boxplot)
# ----------------------------------------------------------------------
def plot_bmi_by_age(df: pd.DataFrame, save_path: str):
    """Purpose: 연령대x성별 BMI 분포 boxplot 생성. Output: PNG 저장"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=df, x="age_group_label", y="bmi", hue="sex_label", order=AGE_ORDER,
                palette=[OKABE_ITO["blue"], OKABE_ITO["orange"]], width=0.7,
                fliersize=1.5, linewidth=0.8, ax=ax)
    ax.axhline(25, color=OKABE_ITO["green"], linestyle="--", linewidth=1.2,
               label="비만 기준 (BMI 25 kg/m²)")
    ax.set_xlabel("연령대 (세)", fontsize=12)
    ax.set_ylabel("체질량지수 BMI (kg/m²)", fontsize=12)
    ax.set_title("연령대별·성별 BMI 분포", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. 대사증후군 구성요소 상관관계 히트맵
# ----------------------------------------------------------------------
def plot_mets_correlation(df: pd.DataFrame, save_path: str):
    """
    Purpose: 허리둘레/혈압/혈당/중성지방/HDL 간 Spearman 상관행렬을 계산하고
             히트맵으로 시각화한다. (변수 대부분이 우측편포이므로 Pearson보다
             순위기반 Spearman을 1차 지표로 채택)
    Output : (corr_df, pval_df) 및 PNG 저장
    """
    mets_vars = ["waist", "sbp", "dbp", "glucose", "tg", "hdl"]
    labels = {"waist": "허리둘레\n(cm)", "sbp": "수축기혈압\n(mmHg)", "dbp": "이완기혈압\n(mmHg)",
              "glucose": "공복혈당\n(mg/dL)", "tg": "중성지방\n(mg/dL)", "hdl": "HDL-C\n(mg/dL)"}
    sub = df[mets_vars].dropna()

    corr = pd.DataFrame(index=mets_vars, columns=mets_vars, dtype=float)
    pval = pd.DataFrame(index=mets_vars, columns=mets_vars, dtype=float)
    for vi in mets_vars:
        for vj in mets_vars:
            r, p = stats.spearmanr(sub[vi], sub[vj])
            corr.loc[vi, vj] = r
            pval.loc[vi, vj] = p

    cb_diverging = mcolors.LinearSegmentedColormap.from_list(
        "cb_div", [OKABE_ITO["blue"], "#FFFFFF", OKABE_ITO["orange"]])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(corr.astype(float), mask=mask, annot=True, fmt=".2f", cmap=cb_diverging,
                vmin=-1, vmax=1, square=True, linewidths=1, linecolor="white",
                cbar_kws={"label": "Spearman's ρ", "shrink": 0.8},
                xticklabels=[labels[v] for v in mets_vars],
                yticklabels=[labels[v] for v in mets_vars], ax=ax)
    ax.set_title("대사증후군 구성요소 간 상관관계 (Spearman)", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return corr, pval


# ----------------------------------------------------------------------
# 6. 흡연상태별 간기능 지표 비교 (Kruskal-Wallis + post-hoc)
# ----------------------------------------------------------------------
def compare_liver_by_smoking(df: pd.DataFrame, save_path: str):
    """
    Purpose: 3개 흡연군(비흡연/과거흡연/현재흡연) 간 AST/ALT/GGT를 비교한다.
             간효소는 심하게 우측편포(이상치 다수)이므로 Kruskal-Wallis를 1차 검정으로,
             유의 시 Bonferroni 보정 pairwise Mann-Whitney U를 사후검정으로 수행한다.
    Output : (kw_df, posthoc_df) 및 boxplot PNG 저장
    """
    liver_vars = ["ast", "alt", "ggt"]
    labels = {"ast": "AST (IU/L)", "alt": "ALT (IU/L)", "ggt": "γ-GTP (IU/L)"}
    smk_order = ["Never", "Former", "Current"]
    smk_kr = {"Never": "비흡연", "Former": "과거흡연", "Current": "현재흡연"}

    d = df.dropna(subset=["smoking_label"] + liver_vars).copy()
    groups = {g: d.loc[d["smoking_label"] == g] for g in smk_order}

    kw_rows, posthoc_rows = [], []
    for v in liver_vars:
        samples = [groups[g][v].values for g in smk_order]
        h_stat, p_kw = stats.kruskal(*samples)
        k, n_total = 3, len(d)
        eta2 = (h_stat - k + 1) / (n_total - k)  # KW 기반 eta-squared 근사치
        kw_rows.append({"Variable": labels[v], "H": round(h_stat, 2), "df": k - 1,
                         "p-value": "<0.001" if p_kw < 0.001 else round(p_kw, 3),
                         "eta-squared": round(eta2, 4)})

        pairs = list(combinations(smk_order, 2))
        for g1, g2 in pairs:
            x, y = groups[g1][v], groups[g2][v]
            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_bonf = min(p * len(pairs), 1.0)
            r = rank_biserial(u, len(x), len(y))
            posthoc_rows.append({
                "Variable": labels[v], "Comparison": f"{smk_kr[g1]} vs {smk_kr[g2]}",
                "Median1": round(x.median(), 1), "Median2": round(y.median(), 1),
                "Bonferroni p": "<0.001" if p_bonf < 0.001 else round(p_bonf, 3),
                "rank-biserial r": round(r, 3),
            })

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    pal = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["purple"]]
    for ax, v in zip(axes, liver_vars):
        sns.boxplot(data=d, x="smoking_label", y=v, order=smk_order, hue="smoking_label",
                    palette=pal, width=0.55, fliersize=1.5, linewidth=0.9, legend=False, ax=ax)
        ax.set_yscale("log")
        ax.set_xticks(range(len(smk_order)))
        ax.set_xticklabels([smk_kr[g] for g in smk_order])
        ax.set_xlabel("")
        ax.set_ylabel(f"{labels[v]} (log scale)", fontsize=11)
        ax.set_title(labels[v], fontsize=12, fontweight="bold")
    fig.suptitle("흡연 상태별 간기능 지표 비교", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(kw_rows), pd.DataFrame(posthoc_rows)


# ----------------------------------------------------------------------
# 메인 실행부
# ----------------------------------------------------------------------
def main():
    setup_plot_style()
    df = load_and_preprocess(CFG.data_path, CFG.encoding)
    data_quality_report(df)

    print("[1] 기술통계")
    desc = descriptive_statistics(df, CONT_VARS)
    print(desc.to_string(index=False))
    desc.to_csv(f"{CFG.output_dir}/1_descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    print("\n[2] 성별 비교")
    sexcomp = compare_by_sex(df, CONT_VARS)
    print(sexcomp.to_string(index=False))
    sexcomp.to_csv(f"{CFG.output_dir}/2_sex_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n[3] 연령대별 BMI 분포 boxplot 생성 중...")
    plot_bmi_by_age(df, f"{CFG.output_dir}/3_bmi_by_age_sex.png")

    print("[4] 대사증후군 구성요소 상관관계 분석 중...")
    corr, pval = plot_mets_correlation(df, f"{CFG.output_dir}/4_mets_correlation.png")
    corr.to_csv(f"{CFG.output_dir}/4_mets_correlation_rho.csv", encoding="utf-8-sig")
    pval.to_csv(f"{CFG.output_dir}/4_mets_correlation_pval.csv", encoding="utf-8-sig")
    print(corr.round(3))

    print("\n[5] 흡연상태별 간기능지표 비교 중...")
    kw_df, posthoc_df = compare_liver_by_smoking(df, f"{CFG.output_dir}/5_liver_by_smoking.png")
    print(kw_df.to_string(index=False))
    print(posthoc_df.to_string(index=False))
    kw_df.to_csv(f"{CFG.output_dir}/5_liver_smoking_kruskal.csv", index=False, encoding="utf-8-sig")
    posthoc_df.to_csv(f"{CFG.output_dir}/5_liver_smoking_posthoc.csv", index=False, encoding="utf-8-sig")

    print(f"\n모든 결과가 '{CFG.output_dir}/' 에 저장되었습니다.")


if __name__ == "__main__":
    main()
