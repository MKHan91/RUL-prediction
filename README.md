<div align="center">
  <p>
    <strong>진동 추세를 분석하고, 고장 임계값에 도달하는 시점을 추정합니다.</strong><br />
    RMS 시계열에 회귀 모델을 적용해 잔여수명 추정에 필요한 정보를 확인하는 Python 프로젝트입니다.
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
    <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy" />
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib" />
    <img src="https://img.shields.io/badge/h5py-Data%20I%2FO-655D4D?style=flat-square" alt="h5py" />
  </p>

</div>

---

## 🔎 프로젝트 개요

설비에서 수집한 진동 데이터는 운전 상태에 따라 달라집니다. 이 변화를 RMS 시계열로 확인하고, 회귀 곡선을 이용해 설정한 고장 임계값에 도달하는 시점을 추정하도록 구현했습니다.

현재 저장소에는 임계값 기반 회귀 추정 코드와 베이지안 선형 회귀 예제 코드가 있습니다.

- **Threshold-based Regression:** RMS 추세를 회귀 곡선으로 근사하고, 고장 임계값에 도달하는 시점을 추정합니다.
- **Bayesian Linear Regression:** 사전분포와 관측 데이터를 이용해 회귀 가중치의 사후분포를 계산하고, 예측값과 불확실성을 다룹니다.

두 방법은 현재 코드에서 별도로 구성되어 있습니다. 베이지안 회귀 예제는 기존 RUL 추정 파이프라인과 통합되어 있지 않습니다.

## ⚙️ RUL 추정 알고리즘

### 1. RMS 시계열 평활화

진동 신호에서 계산한 RMS 시계열에 이동평균을 적용해 단기적인 변동을 줄입니다.

이동평균 윈도우 크기에 따라 추세가 달라지므로, 이후 회귀 곡선과 임계값 도달 시점에도 영향을 줍니다.

### 2. 회귀 모델 적합

현재 시점 이전의 최근 관측 구간을 사용해 회귀 모델을 적합합니다.

선형 모델과 여러 비선형 모델을 선택할 수 있으며, 모델마다 표현하는 추세의 형태가 다릅니다.

### 3. 미래 구간 외삽

적합된 회귀 곡선을 미래 구간으로 확장해 RMS의 변화를 추정합니다.

### 4. 고장 임계값 도달 시점 계산

예측된 곡선이 설정한 고장 발생 임계값을 처음 초과하는 인덱스를 찾습니다.

```text
RMS 데이터
    ↓
이동평균 평활화
    ↓
최근 관측 구간 선택
    ↓
회귀 모델 적합
    ↓
미래 구간 외삽
    ↓
고장 임계값 최초 통과 인덱스 탐색
    ↓
추정 결과 시각화
```

현재 코드는 임계값을 통과하는 시점을 `Predicted RUL`로 표시합니다. 다만 계산 결과는 현재 시점으로부터 남은 시간이 아니라 임계값 도달 시점의 절대 인덱스에 가깝습니다.

## 📈 지원 회귀 모델

`RUL_regression_v3.py`에서 모델을 선택할 수 있으며, 비선형 회귀 함수는 `NonlinearRegression.py`에 정의되어 있습니다.

| 모델 | 회귀식 | 구현 함수 |
|---|---|---|
| Linear | 선형 추세 | `LinearRegression` |
| Exponential | \(y = ae^{bx}\) | `fit_exp()` |
| Logarithmic | \(y = a\ln(x) + b\) | `fit_log()` |
| Lloyd–Lipow | \(y = a - b/x\) | `fit_lipow()` |
| Gompertz | \(y = ae^{-be^{-cx}}\) | `fit_gom()` |
| Power | \(y = ax^b\) | `fit_pow()` |

각 모델의 적합 결과와 미래 추세를 확인할 수 있습니다. 어떤 모델이 적합한지는 RMS 변화와 실제 고장 이력을 기준으로 검토해야 합니다.

## 🧮 베이지안 선형 회귀

`BLR_ex.py`에는 베이지안 선형 회귀 계산 예제가 별도로 포함되어 있습니다.

일반적인 회귀 모델이 관측 데이터에 가장 잘 맞는 파라미터를 추정하는 방식이라면, 베이지안 회귀는 파라미터에 대한 사전분포와 관측 데이터를 이용해 사후분포를 계산합니다.

### 알고리즘

**1. 사전분포 설정**

회귀 가중치 $w$에 정규분포를 가정합니다.

$$
p(w)=\mathcal{N}(w\mid m_0,S_0)
$$

여기서 $m_0$는 사전 평균, $S_0$는 사전 공분산입니다.

**2. 관측 데이터 반영**

입력 데이터에서 설계행렬 $\Phi$를 구성하고, 관측값 $t$를 이용해 가중치의 사후분포를 계산합니다.

$$
p(w\mid t)=\mathcal{N}(w\mid m_N,S_N)
$$

사후분포는 기존 사전분포와 관측 데이터의 정보를 함께 반영합니다.

**3. 예측값과 불확실성 계산**

추정한 사후분포를 바탕으로 새로운 입력에 대한 예측값과 예측 불확실성을 계산합니다.

```text
입력 데이터
    ↓
Basis Function으로 설계행렬 구성
    ↓
사전분포 설정
    ↓
관측 데이터 반영
    ↓
가중치 사후분포 계산
    ↓
예측값 및 예측 불확실성 계산
```

`BLR_ex.py`에서는 `posterior_general()`로 사후분포를 계산하고, `posterior_predictive()`로 예측값과 불확실성을 계산합니다.

다만 현재 예제는 소규모 배열을 사용하는 별도 스크립트입니다. RMS 데이터를 이용해 RUL을 추정하거나, 예측 불확실성을 고장 시점 계산에 반영하는 단계까지 연결되어 있지는 않습니다.

## 🗂️ 프로젝트 구조

```text
RUL-prediction/
├── RUL_regression.py        # 초기 회귀 기반 추정
├── RUL_regression_v2.py     # 회귀 로직 확장 버전
├── RUL_regression_v3.py     # 대화형 실행 및 RUL 시각화
├── NonlinearRegression.py   # 비선형 회귀 함수
├── BLR_ex.py                # 베이지안 선형 회귀 예제
├── Plotting.py              # RMS 데이터 시각화
└── back_data/               # 보조 코드 및 데이터 유틸리티
```

## 🧰 환경 설정

다음 Python 패키지를 설치합니다.

```bash
python -m pip install numpy scipy scikit-learn matplotlib h5py
```

단, `BLR_ex.py`는 별도의 `Bayesian_approach` 모듈도 import하므로 해당 모듈의 의존성 및 경로를 별도로 확인해야 합니다.

## ▶️ 실행 방법

Windows 환경을 기준으로 합니다.

1. `h5py`로 읽을 수 있는 `.mat` 파일을 데이터 폴더에 넣습니다. 파일명에는 `_rms`가 포함되어야 합니다.
2. 저장소 디렉터리에서 실행합니다.

   ```powershell
   python RUL_regression_v3.py
   ```

3. 데이터 폴더 경로를 입력합니다. 빈 값으로 Enter를 누르면 스크립트에서 지정한 기본 `data` 폴더를 사용합니다.
4. 이동평균 크기, 회귀 모델, 시간 간격, 적합 윈도우, 예측 범위 및 임계값을 설정합니다.

### 주요 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| 데이터 경로 | `./data` | RMS 데이터 폴더 |
| 이동평균 크기 | `45` | 평활화 윈도우 크기 |
| 회귀 모델 | `linear` | 적용할 회귀 모델 |
| 시간 간격 | `5` | 현재 시점을 이동하는 간격 |
| 적합 윈도우 | `15` | 회귀에 사용할 최근 샘플 수 |
| 예측 범위 | `500` | 미래 방향으로 평가할 인덱스 수 |
| 안전 임계값 | `1.45` | 안전 상태 참고값 |
| 고장 임계값 | `2.2` | 고장 상태 판정 기준 |
| 고장 발생 임계값 | `9.0` | 통과 시점을 추정할 최종 기준 |

위 값은 코드에 설정된 기본값이며, 모든 설비에 적용할 수 있는 권장값은 아닙니다.

## 🖥️ 결과 시각화

Matplotlib 그래프에서 다음 항목을 확인할 수 있습니다.

- RMS 관측값과 평활화된 추세
- 선택한 회귀 모델의 적합 곡선
- 현재 시점과 회귀 적합 구간
- 안전·고장·고장 발생 임계값
- 임계값에 도달하는 것으로 추정된 인덱스

현재 구현은 결과를 화면에 표시하며, 결과 이미지를 자동으로 저장하지는 않습니다.

## ⚠️ 구현상 확인할 점

- **데이터 미포함:** 호환되는 RMS `.mat` 데이터는 별도로 준비해야 합니다.
- **첫 번째 시계열 사용:** 데이터 로더가 여러 파일을 읽더라도 현재 `main()`은 `rms_data[0]`을 사용합니다.
- **RUL 정의:** 임계값 도달 인덱스와 현재 인덱스의 차이를 계산해야 실제 잔여 시간 또는 잔여 사이클에 가까운 값을 얻을 수 있습니다.
- **임계값 검증:** 이동평균 크기, 회귀 모델 및 임계값은 설비의 고장 이력과 비교해 검증해야 합니다.
- **베이지안 코드의 범위:** `BLR_ex.py`는 별도 예제이며 현재 임계값 기반 RUL 추정 흐름에는 연결되어 있지 않습니다.
- **실행 환경:** v3 코드에 Windows 방식의 경로 처리와 `os.system('pause')` 호출이 있어 Linux/macOS에서는 수정이 필요할 수 있습니다.

---
