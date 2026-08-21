# 10년 넘게 멈춰 있던 FDTD 프로젝트, 어디서부터 정리해야 할까

> 작성일: 2026-08-21  
> 대상 프로젝트: GMES(GIST Maxwell's Equations Solver)

GMES는 명시적 유한차분 시간영역법(FDTD)으로 맥스웰 방정식을 푸는 전자기 시뮬레이터다. Python API 위에 C++, SWIG, Cython 확장을 결합한 구조로, 1·2·3차원 시뮬레이션과 여러 분산 물질 모델, PML 경계 조건, MPI 병렬 실행 등을 지원한다.

하지만 핵심 코드와 테스트의 상당 부분은 2010년대 초반 환경에 머물러 있다. 이번 점검의 목적은 기존 Python 2 호환성을 보존하는 것이 아니라, 최신 안정 Python으로 옮기기 전에 어떤 관리 작업이 필요한지 파악하는 것이었다.

결론부터 말하면, 이 프로젝트에는 단순한 저장소 정리가 아니라 하나의 **현대화 릴리스**가 필요하다.

## 목표 환경

2026년 8월 현재 최신 안정 버전은 Python 3.14.7이다. Python 3.15는 아직 시험판이며 정식 출시는 2026년 10월로 예정되어 있다. 따라서 우선 Python 3.14를 공식 지원 대상으로 삼고, Python 3.15는 사전 호환성 검사 대상으로 두는 것이 적절하다.

- [Python 공식 릴리스 현황](https://www.python.org/downloads/)
- [Python 3.14.7 릴리스 정보](https://www.python.org/downloads/release/python-3147/)

최신 환경의 기본 조합은 다음과 같이 잡을 수 있다.

- Python 3.14
- NumPy 2
- Cython 3
- SWIG 4
- 현대 C++ 컴파일러
- `pyproject.toml` 기반 격리 빌드

## 우선순위별 관리 작업

| 우선순위 | 작업 | 필요한 이유 |
| --- | --- | --- |
| P0 | Python 3.14 마이그레이션 | 현재 Python 3에서 패키지를 import할 수 없다. |
| P0 | 빌드·패키징 현대화 | 기존 빌드가 제거된 Distutils에 의존한다. |
| P0 | 수치 회귀 테스트 확보 | Python 2와 3의 의미 차이가 계산 결과를 바꿀 수 있다. |
| P0 | SWIG·Cython·NumPy 바인딩 갱신 | 2009년 전후의 Python/NumPy C API를 사용한다. |
| P1 | CI와 재현 가능한 개발 환경 구축 | 자동 빌드와 테스트가 전혀 없다. |
| P1 | 테스트 구조와 범위 개선 | 테스트가 일부 재료 모델에 편중되어 있다. |
| P1 | MPI·시각화·HDF5 기능 정비 | 선택 기능이 폐기되거나 변경된 API를 사용한다. |
| P2 | 문서·릴리스·저장소 위생 개선 | 현재 문서와 프로젝트 정책이 현대화 목표와 충돌한다. |

## 1. Python 3에서는 아직 시작조차 할 수 없다

Python 3.14 환경에서 소스 전체를 컴파일해 보니 다음 모듈에서 즉시 오류가 발생했다.

- `gmes/fdtd.py`: Python 2 방식 `print` 문
- `gmes/geometry.py`: Python 2 방식 `print` 문
- `gmes/source.py`: Python 2 방식 `print` 문과 구형 예외 문법
- `gmes/show.py`: 탭과 공백이 섞인 들여쓰기 오류
- 일부 예제와 유틸리티: Python 2 전용 문법

정적 검색에서 확인된 대표적인 변환 대상은 다음과 같다.

| 패턴 | 확인된 수량 |
| --- | ---: |
| Python 2 방식 `print` | 228개 |
| `xrange` | 17개 |
| `iteritems`, `itervalues` 등 | 15개 |
| `has_key` | 15개 |
| `generator.next()` | 3개 |
| 제거된 `np.int` | 17개 |

패키지 내부 import도 Python 2의 암시적 상대 import에 의존한다. 예를 들어 `gmes/__init__.py`는 `from fdtd import *` 형태를 사용하는데, Python 3에서는 `from .fdtd import *`처럼 패키지 상대 경로를 명시해야 한다.

이 밖에도 다음 항목을 함께 정리해야 한다.

- `collections.Sequence`를 `collections.abc.Sequence`로 변경
- 리스트처럼 사용되는 `map` 결과를 명시적으로 변환
- `psyco` 관련 최적화 코드 제거
- `scipy.sqrt` 같은 오래전에 제거된 API 교체
- Python 2 전용 shebang과 실행 방법 갱신

## 2. 문법보다 위험한 것은 수치 의미의 변화다

Python 2 코드를 자동 변환하는 것만으로는 충분하지 않다. 과학 계산 코드에서는 문법이 맞더라도 계산 의미가 달라질 수 있기 때문이다.

가장 대표적인 사례는 나눗셈이다. `gmes/geometry.py`의 격자와 MPI 영역 분할 코드는 Python 2의 정수 나눗셈에 기대는 부분이 있다.

```python
self.general_field_size = \
    self.whole_field_size / self.cart_comm.topo[0]
```

Python 3에서 `/`는 항상 실수 나눗셈이다. 이 값을 기계적으로 이식하면 배열 크기가 정수에서 실수로 바뀌거나, 반올림 방식이 달라져 영역 분할 결과가 변할 수 있다. 필요한 곳에는 `//`를 사용하고, 나머지 분배와 경계 셀 처리 규칙을 테스트로 명시해야 한다.

따라서 마이그레이션 전에 다음과 같은 기준 결과를 확보해야 한다.

- 진공에서의 파동 전파 속도
- Fresnel 반사율과 투과율
- 에너지 보존 또는 알려진 감쇠 특성
- PML 경계의 반사 오차
- 실수장과 복소수장의 동일 조건 비교
- 단일 프로세스와 MPI 분할 결과 비교

기존 Python 2 환경을 제품으로 계속 지원할 필요는 없지만, 가능한 경우 격리된 과거 환경에서 기준값을 한 번 추출할 가치는 있다. 그것이 어렵다면 분석해나 다른 검증된 시뮬레이터 결과를 기준으로 삼아야 한다.

## 3. 빌드 시스템은 교체가 필요하다

현재 `setup.py`는 표준 라이브러리의 `distutils.core`를 직접 가져온다. Distutils는 Python 3.12에서 표준 라이브러리에서 제거되었기 때문에 Python 3.14용 공식 빌드 방식으로 사용할 수 없다.

- [Python Distutils 제거 안내](https://docs.python.org/3/library/distutils.html)

권장되는 정리 방향은 다음과 같다.

1. `pyproject.toml`에 빌드 백엔드와 빌드 의존성을 선언한다.
2. `setuptools.Extension`과 Cython 빌드를 사용하도록 전환한다.
3. NumPy 헤더가 격리 빌드 환경에 설치되도록 설정한다.
4. 필수 런타임 의존성과 선택 기능을 분리한다.
5. 소스 배포본과 wheel을 모두 검증한다.

선택 기능은 extras로 나누는 편이 좋다.

```text
gmes[plot]   -> Matplotlib
gmes[mpi]    -> mpi4py
gmes[hdf5]   -> PyTables 또는 h5py
gmes[all]    -> 모든 선택 기능
```

현재는 NumPy, SciPy, Cython, Matplotlib, mpi4py, PyTables의 버전이나 필수 여부가 패키지 메타데이터에 선언되어 있지 않다. 이 상태에서는 사용자가 동일한 환경을 재현할 수도, 의존성 보안 검사를 자동화할 수도 없다.

## 4. 가장 큰 기술적 난점은 네이티브 바인딩이다

순수 C++ 소스는 현재 Apple Clang에서 C++20 문법 검사를 통과했다. 즉, 초기 난점은 계산 코어 자체보다 Python과 C++ 사이의 경계에 집중되어 있다.

특히 `src/numpy.i`는 2009년 무렵의 NumPy SWIG typemap으로 보이며 다음과 같은 API를 사용한다.

- `PyString_Check`
- `PyInt_Check`
- `PyFile_Check`
- `PyInstance_Check`
- `PyArrayObject` 내부 필드 직접 접근

이 API들은 최신 Python 또는 NumPy에서 그대로 사용할 수 없다. NumPy 2에서는 일부 C 구조체와 API의 가시성 및 사용 방식도 변경됐다.

- [NumPy 2.0 마이그레이션 가이드](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [NumPy C API 문서](https://numpy.org/doc/stable/reference/c-api/)

이 파일은 부분적으로 땜질하기보다 현재 NumPy C API에 맞는 typemap 또는 더 작은 전용 변환 계층으로 교체하는 것이 안전하다.

Cython 소스도 Python 3 의미론을 명시하고 수정해야 한다. Cython 3은 기본적으로 Python 3 의미론을 적용하므로, 기존 코드의 `print`, `xrange`, `map`, 나눗셈 동작이 한꺼번에 달라진다.

- [Cython 3 마이그레이션 가이드](https://cython.readthedocs.io/en/latest/src/userguide/migrating_to_cy30.html)

## 5. 테스트가 존재하지만 안전망으로는 부족하다

저장소에는 8개의 `unittest` 파일이 있다. 그러나 모두 pointwise material 계열에 집중되어 있다.

직접적인 회귀 테스트가 부족한 영역은 다음과 같다.

- 전체 FDTD 시간 진행
- geometry와 좌표 변환
- source 파형과 배치
- UPML·CPML 경계의 실제 반사 성능
- 파일 입출력
- 시각화
- MPI 영역 분할과 통신
- 예제의 최소 실행 경로

기존 테스트 자체에도 개선할 점이 있다.

- 난수를 seed 없이 사용한다.
- `ex = hz = hy = np.zeros(...)`처럼 서로 다른 필드가 같은 배열을 공유하는 패턴이 108개 있다.
- 저장소 루트를 `sys.path`에 수동으로 추가한다.
- 기본 테스트 탐색 규칙과 파일명이 맞지 않는다.
- 전체 테스트를 실행하는 단일 명령이 없다.

특히 여러 필드가 같은 배열을 공유하면 한 필드를 갱신했을 때 다른 필드까지 함께 바뀐다. 실제 시뮬레이션과 다른 조건이므로 오류를 발견하기는커녕 감출 가능성이 있다.

테스트 정비 시에는 다음 원칙이 필요하다.

- 모든 입력을 결정론적으로 구성
- 각 필드에 독립적인 배열 사용
- 작은 격자와 짧은 시간 구간을 기본 smoke test로 사용
- 값의 정확한 일치보다 물리적으로 타당한 허용 오차 사용
- 무거운 예제와 빠른 단위 테스트 분리
- coverage를 측정하되 수치 검증의 질을 우선

## 6. 선택 기능과 운영 코드도 별도로 점검해야 한다

핵심 계산 외의 코드에도 즉시 드러나는 문제가 있다.

- `gmes/file_io.py`의 HDF5 함수는 `openFile` import가 주석 처리되어 호출 시 실패한다.
- PyTables의 구형 camelCase API를 사용한다.
- Matplotlib backend를 `TkAgg`로 강제하여 headless CI 환경과 충돌할 수 있다.
- MPI communicator의 속성과 호출 방식이 현대 mpi4py와 맞는지 검증되지 않았다.
- `utils/dia_util.py`는 오래된 Dia 연동과 다수의 `eval` 호출을 포함한다.

이 기능들은 무조건 모두 이식하기보다 실제 사용 여부를 먼저 판단하는 것이 좋다. 사용되지 않는 도구는 `legacy/`로 이동하거나 제거하고, 유지할 기능만 테스트와 함께 현대화해야 한다.

## 7. 자동화와 릴리스 관리가 없다

현재 GitHub 설정에는 Sponsors 파일만 있고 CI 워크플로는 없다. 다음 자동화가 필요하다.

- Python 3.14에서 소스 빌드
- Cython·SWIG 확장 컴파일
- 단위 테스트와 수치 회귀 테스트
- Linux와 macOS 빌드
- wheel 생성 및 설치 검사
- 정적 검사와 포맷 검사
- 의존성 업데이트 검사
- Python 3.15 시험판 사전 호환성 검사

Python 3.15 작업은 초기에는 실패를 허용하되, 정식 출시 전에 필수 검사로 전환하면 된다.

릴리스 관리 측면에서도 다음 항목이 빠져 있다.

- 현대적인 버전 정책
- changelog
- 배포 절차
- wheel 지원 범위
- 지원 운영체제와 컴파일러 명시
- 보안 문제 신고 절차

현재 버전 `0.9.5` 이후 현대화 릴리스를 `0.10.0`으로 이어갈지, 호환성 단절을 반영해 새로운 주 버전을 사용할지는 구현 범위를 확정한 뒤 결정해야 한다.

## 권장 실행 순서

### 1단계: 기준과 안전망 만들기

- Python 3.14를 공식 목표로 선언
- 지원 운영체제와 컴파일러 결정
- 대표적인 수치 기준값 확보
- 빠른 smoke test 설계

### 2단계: 빌드와 import 복구

- `pyproject.toml` 추가
- 의존성과 optional extras 선언
- 패키지 내부 상대 import 수정
- Python 3 문법과 기본 자료형 API 변환
- `python -m build`와 wheel 설치 검증

### 3단계: 네이티브 확장 현대화

- Cython 3 의미론 적용
- SWIG 4에서 wrapper 재생성
- NumPy 2 C API에 맞게 typemap 교체
- 실수·복소수 배열과 메모리 소유권 테스트

### 4단계: 수치 기능 검증

- geometry와 좌표 변환 테스트
- source와 경계 조건 테스트
- 전체 FDTD 회귀 테스트
- MPI 단일·다중 프로세스 결과 비교

### 5단계: 운영 체계 완성

- GitHub Actions CI
- wheel 빌드와 릴리스 자동화
- README와 기여 지침 갱신
- changelog, 보안 정책, Dependabot 추가
- 사용하지 않는 유틸리티와 예제 정리

## 마무리

이 프로젝트의 현대화에서 가장 중요한 원칙은 “Python 3에서 실행되게 만드는 것”과 “계산 결과가 계속 옳은 것”을 분리하지 않는 것이다.

문법 변환과 최신 빌드 도구 도입은 비교적 기계적으로 진행할 수 있다. 반면 정수 나눗셈, 배열 dtype, iterator 동작, C API, MPI 영역 분할은 시뮬레이션 결과를 조용히 바꿀 수 있다. 먼저 수치 안전망을 만들고, 빌드 계층부터 한 단계씩 복구하는 것이 가장 안전한 접근이다.

첫 번째 구현 이정표는 다음처럼 잡는 것이 적절하다.

> **Python 3.14에서 격리 빌드가 성공하고, 패키지를 import할 수 있으며, 작은 결정론적 시뮬레이션 하나가 검증된 결과를 내는 상태**

여기까지 도달하면 이후의 기능별 마이그레이션과 성능 개선은 훨씬 예측 가능한 작업이 된다.
