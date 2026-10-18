<div align="center">

  <p>
    <strong>진동 추세를 분석하고, 고장 임계값에 도달하는 시점을 추정합니다.</strong><br />
    RMS 시계열에 회귀 곡선을 적용해 잔여수명 추정에 필요한 정보를 확인하는 Python 프로젝트입니다.
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

처리 과정은 다음과 같습니다.

1. MATLAB `.mat` 파일에서 RMS 데이터 로드
2. 이동평균으로 시계열 평활화
3. 최근 관측 구간에 회귀 곡선 적합
4. 예측 곡선이 고장 임계값에 도달하는 시점 계산
5. RMS 추세와 예측 결과 시각화

딥러닝 모델 대신 선형·비선형 회귀식을 사용합니다. 각 모델의 추세를 비교하고, 임계값과의 관계를 그래프에서 확인할 수 있도록 구성했습니다.

> 현재 구현은 회귀 기반 RUL 추정 프로토타입입니다. 실제 설비의 잔여수명을 예측하는 성능이 검증된 시스템은 아닙니다.

## 🧰 환경 설정

다음 Python 패키지를 설치합니다.

```bash
python -m pip install numpy scipy scikit-learn matplotlib h5py
```

## ▶️ 실행 방법
Windows 환경을 기준으로 합니다.

1. `h5py`로 읽을 수 있는 `.mat` 파일을 데이터 폴더에 넣습니다. 파일명에 `_rms`가 포함되어야 합니다. 예: `machine_rms.mat`
2. 저장소 디렉터리에서 실행합니다.

   ```powershell
   python RUL_regression_v3.py
   ```

3. 데이터 폴더 경로를 입력합니다. 빈 값으로 Enter를 누르면 Python 파일 옆의 기본 `data` 폴더를 사용합니다.
4. 입력 프롬프트에서 평활화 윈도우, 회귀 모델, 시간 간격, 적합 윈도우, 예측 범위, 임계값을 설정합니다. 표시된 기본값을 사용하려면 Enter를 누릅니다.


## ⚙️ 처리 흐름

<div align="center">

| 01. 데이터 로드 | 02. 신호 평활화 | 03. 회귀 곡선 적합 | 04. 고장 시점 추정 |
|:---:|:---:|:---:|:---:|
| RMS `.mat` | 이동평균 | 최근 관측 구간 | 임계값 통과 시점 |
| 데이터 확인 | 단기 변동 완화 | 추세 근사 | 미래 구간 외삽 |

</div>

### 1. 데이터 로드

`h5py`로 읽을 수 있는 MATLAB `.mat` 파일 중 파일명에 `_rms`가 포함된 데이터를 불러옵니다.

### 2. 신호 평활화

이동평균을 적용해 RMS 시계열의 단기 변동을 줄입니다. 윈도우 크기에 따라 평활화 결과가 달라집니다.

### 3. 회귀 곡선 적합

선택한 회귀 모델을 최근 관측 구간에 적용합니다. 적합에 사용하는 샘플 수와 모델에 따라 추세 곡선이 달라집니다.

### 4. 고장 시점 추정

회귀 곡선을 미래 구간으로 외삽하고, 설정한 임계값을 통과하는 인덱스를 찾습니다.

### 5. 결과 확인

Matplotlib 그래프에서 관측 RMS, 회귀 곡선, 현재 시점, 적합 구간, 상태별 임계값을 함께 확인합니다.

## 📈 회귀 모델

`RUL_regression_v3.py`에서 회귀 모델을 선택할 수 있습니다. 비선형 모델의 함수는 `NonlinearRegression.py`에 정의되어 있습니다.

| 모델 | 회귀식 | 구현 |
|---|---|---|
| Linear | 선형 추세 | `LinearRegression` |
| Exponential | \(y = ae^{bx}\) | `fit_exp()` |
| Logarithmic | \(y = a\ln(x) + b\) | `fit_log()` |
| Lloyd–Lipow | \(y = a - b/x\) | `fit_lipow()` |
| Gompertz | \(y = ae^{-be^{-cx}}\) | `fit_gom()` |
| Power | \(y = ax^b\) | `fit_pow()` |

실제 적용 시에는 데이터의 추세와 고장 이력을 기준으로 모델 및 임계값을 확인해야 합니다.

## 🗂️ 프로젝트 구조

```text
RUL-prediction/
├── RUL_regression.py        # 초기 회귀 기반 추정 코드
├── RUL_regression_v2.py     # 회귀 로직 확장 버전
├── RUL_regression_v3.py     # 대화형 실행 및 시각화
├── NonlinearRegression.py   # 비선형 회귀 함수
├── Plotting.py              # RMS 데이터 시각화
├── BLR_ex.py                # 베이지안 선형 회귀 예제
└── back_data/               # 보조 코드 및 데이터 유틸리티
