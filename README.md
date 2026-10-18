<div align="center">
  <p>
    <strong>진동 추세에서 고장 시점 추정까지.</strong><br />
    RMS 시계열과 회귀 곡선을 이용해 고장 임계값 도달 시점을 추정하는 Python 프로토타입입니다.
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

## 🔎 프로젝트 소개

**RUL-prediction**은 설비 상태 모니터링 데이터에서 잔여수명(Remaining Useful Life, RUL)을 추정하기 위한 임계값 기반 회귀 접근법을 다룹니다. 현재 구현은 RMS 시계열을 이동평균으로 평활화한 다음, 최근 관측 구간에 회귀 곡선을 맞추고, 예측 곡선이 고장 임계값을 통과하는 시간 인덱스를 추정합니다.

딥러닝 모델이 아니라 해석하기 쉬운 회귀 곡선을 사용합니다. 관측 신호와 예측 추세, 상태별 임계값을 Matplotlib으로 시각화해 추정 근거를 확인할 수 있습니다.

## 🎯 프로젝트 핵심

**해석 가능한 임계값 기반 설비 상태 모니터링**의 예제입니다. 평활화된 열화 신호를 추세 곡선으로 근사하고, 설정한 고장 임계값에 도달하는 시간 인덱스를 시각적으로 추정합니다. 검증된 상용 RUL 시스템이 아니라 연구·프로토타이핑 입니다.


## ⚙️ 처리 흐름

<div align="center">
  <table>
    <tr>
      <td align="center"><strong>01 · 데이터 로드</strong><br /><code>.mat</code>의 RMS 값</td>
      <td align="center">→</td>
      <td align="center"><strong>02 · 평활화</strong><br />이동평균 필터</td>
      <td align="center">→</td>
      <td align="center"><strong>03 · 회귀</strong><br />최근 구간에 곡선 적합</td>
      <td align="center">→</td>
      <td align="center"><strong>04 · 시점 추정</strong><br />고장 임계값 통과</td>
    </tr>
  </table>
</div>

1. **RMS 데이터 로드:** `h5py`로 읽을 수 있고 파일명에 `_rms`가 포함된 MATLAB `.mat` 파일을 읽습니다.
2. **신호 평활화:** 이동평균으로 단기 변동을 줄입니다.
3. **추세 곡선 적합:** 최근 관측 윈도우에 회귀 모델을 맞춥니다.
4. **미래 구간 외삽:** 회귀 곡선이 설정된 고장 임계값에 도달하는 위치를 계산합니다.
5. **결과 시각화:** 관측 RMS, 추세선, 현재 시점과 윈도우 표시, 안전·고장·고장 임계값을 함께 표시합니다.

## 📈 지원 회귀 모델

`RUL_regression_v3.py`는 `NonlinearRegression.py`의 함수를 이용해 다음 모델을 지원합니다.

| 모델 | 곡선 | 구현 함수 |
|---|---|---|
| 선형 | 선형 추세 | `sklearn.linear_model.LinearRegression` |
| 지수 | \(y = a e^{bx}\) | `fit_exp()` |
| 로그 | \(y = a\ln(x) + b\) | `fit_log()` |
| Lloyd–Lipow | \(y = a - b/x\) | `fit_lipow()` |
| Gompertz | \(y = a e^{-b e^{-cx}}\) | `fit_gom()` |
| 거듭제곱 | \(y = ax^b\) | `fit_pow()` |

프로그램 실행 중 회귀 모델을 선택할 수 있습니다. 적절한 곡선과 임계값은 설비 및 신호 특성에 따라 달라지므로, 이 저장소는 범용적인 정확도는 아닙니다.

## 🗂️ 저장소 구조

```text
RUL-prediction/
├── RUL_regression.py       # 초기 임계값 기반 회귀 파이프라인
├── RUL_regression_v2.py    # 회귀 로직을 확장한 중간 버전
├── RUL_regression_v3.py    # 대화형 실행 및 시각화
├── NonlinearRegression.py  # 비선형 곡선 정의 및 파라미터 추정
├── Plotting.py             # 조건에 맞는 .mat RMS 데이터 시각화
├── BLR_ex.py               # 베이지안 선형 회귀 예제
└── back_data/              # 보조 코드 및 데이터 유틸리티
```

## 🧰 환경 설정

다음 Python 패키지를 설치합니다.

```bash
python -m pip install numpy scipy scikit-learn matplotlib h5py
```

## ▶️ 실행 방법

현재 스크립트는 Windows 방식의 경로 구분자와 콘솔 `pause` 명령을 사용하므로, 기본 흐름은 Windows 환경을 기준으로 합니다.

1. `h5py`로 읽을 수 있는 `.mat` 파일을 데이터 폴더에 넣습니다. 파일명에 `_rms`가 포함되어야 합니다. 예: `machine_rms.mat`
2. 저장소 디렉터리에서 실행합니다.

   ```powershell
   python RUL_regression_v3.py
   ```

3. 데이터 폴더 경로를 입력합니다. 빈 값으로 Enter를 누르면 Python 파일 옆의 기본 `data` 폴더를 사용합니다.
4. 입력 프롬프트에서 평활화 윈도우, 회귀 모델, 시간 간격, 적합 윈도우, 예측 범위, 임계값을 설정합니다. 표시된 기본값을 사용하려면 Enter를 누릅니다.

### 주요 파라미터

| 파라미터 | 코드 기본값 | 의미 |
|---|---:|---|
| 데이터 경로 | `./data` | `.mat` 입력 파일이 있는 폴더 |
| 이동평균 크기 | `45` | RMS 평활화에 사용하는 샘플 수 |
| 회귀 모델 | `linear` | `linear`, `exponential`, `power`, `log`, `gompertz`, `Lloyd-Lipow` |
| 시간 간격 | `5` | 검사할 현재 시점 사이의 간격 |
| 적합 윈도우 | `15` | 회귀에 사용하는 최근 샘플 수 |
| 예측 범위 | `500` | 미래 방향으로 평가할 인덱스 수 |
| 안전 임계값 | `1.45` | 안전 상태 참고값 |
| 고장 임계값 | `2.2` | 최근 구간이 고장 상태인지 판단하는 기준값 |
| 고장 발생 임계값 | `9.0` | 통과 시점을 추정하는 최종 임계값 |
| 플롯 갱신 지연 | `0.05` | 플롯을 순차적으로 보여줄 때의 지연 시간 |

위 값은 `RUL_regression_v3.py`에서 확인되는 기본값이며 모든 설비에 권장되는 임계값은 아닙니다.

## 🖥️ 출력 결과

Matplotlib 대화형 플롯이 열립니다. RMS 추세, 관측 구간, 회귀 외삽 곡선, 현재 시점 및 적합 윈도우, 안전·고장·고장 발생 임계값을 확인할 수 있습니다. 현재 활성화된 흐름에서는 결과 이미지를 자동으로 저장하지 않고 화면에 표시합니다.

## ⚠️ 참고 사항 및 한계

- **데이터 미포함:** 호환되는 RMS `.mat` 데이터는 별도로 준비해야 합니다. 파일명에 `_rms`가 포함되어야 하며 `h5py`가 읽을 수 있어야 합니다.
- **현재는 첫 번째 시계열만 사용:** 로더는 여러 파일을 읽지만, `main()`은 `rms_data[0]`을 사용합니다. 여러 설비를 일괄 평가하는 기능은 이 버전에 구현되어 있지 않습니다.
- **출력값의 의미 확인 필요:** 코드에서는 예측 구간의 회귀 곡선이 임계값을 넘는 인덱스를 찾고 이를 `Predicted RUL`로 표기합니다. 계산식은 예측 구간의 시작 인덱스를 더하므로, 값은 잔여 시간보다는 절대적인 임계값 도달 시점 인덱스에 가깝습니다. 현재 시점으로부터 남은 시간(duration)이 필요하다면 정의를 확인하고 현재 인덱스를 빼야 합니다.
- **예측 범위 내에서 임계값을 넘지 않을 수 있음:** 이 경우 추정값이 없을 수 있으므로 자동화에 사용하기 전 예외 처리가 필요합니다.
- **임계값은 설비별로 검증해야 함:** 알려진 고장 이력으로 이동평균 크기, 적합 윈도우, 회귀 모델과 임계값을 검증한 뒤 결과를 해석해야 합니다.
- **경로 관련 주의:** v3 데이터 로더는 Windows식 경로 구분자를 사용하며 `os.system('pause')`를 호출합니다. Linux/macOS에서 사용하려면 관련 코드를 수정해야 할 수 있습니다.

---
