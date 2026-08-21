# Python으로 시작하는 FDTD 전자기 시뮬레이션: GMES 예제 따라잡기

전자기 시뮬레이션을 처음 시작할 때 가장 어려운 부분은 맥스웰 방정식 자체보다도 계산 영역, 재료, 경계 조건, 소스를 코드로 어떻게 옮겨야 하는지 파악하는 일이다. GMES(GIST Maxwell's Equations Solver)는 이런 요소를 Python 객체로 조립해 FDTD(Finite-Difference Time-Domain) 시뮬레이션을 구성한다.

이 글에서는 GMES 저장소의 [`examples/`](../../examples/)를 바탕으로 가장 단순한 2차원 파동부터 도파관, 평면파 산란, 광결정과 플라즈모닉 구조까지 살펴본다. 모든 예제의 결과를 세세하게 해석하기보다는, 예제를 읽고 자신의 문제로 확장하는 데 필요한 공통 문법과 선택 기준에 초점을 맞춘다.

## GMES와 FDTD

FDTD는 공간과 시간을 격자로 나누고 전기장과 자기장을 번갈아 갱신하는 방법이다. 시간 영역에서 직접 계산하므로 한 번의 시뮬레이션으로 파동의 전파, 반사, 간섭, 산란을 관찰할 수 있다.

GMES는 계산량이 큰 부분을 C++, SWIG, Cython 확장으로 처리하고 시뮬레이션 구성은 Python API로 제공한다. 현재 개발 버전은 Python 3.14 이상, C++17 컴파일러, SWIG 4, NumPy 2를 요구한다.

개발 환경에서는 저장소 루트에서 다음과 같이 설치한다.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,hdf5,plot]"
```

예제는 패키지를 설치한 뒤 저장소 루트에서 실행한다.

```sh
python examples/air2d.py
```

그래픽 환경이 없는 서버라면 Matplotlib의 비대화형 백엔드를 사용할 수 있다.

```sh
MPLBACKEND=Agg python examples/air2d.py
```

## 모든 예제를 관통하는 다섯 단계

GMES 예제는 구조가 거의 같다.

1. `Cartesian`으로 계산 영역과 해상도를 정한다.
2. `DefaultMedium`, `Block`, `Cylinder`, `Sphere` 등으로 재료와 형상을 배치한다.
3. `PointSource`나 `TotalFieldScatteredField`로 파동을 입력한다.
4. 문제에 맞는 FDTD 객체를 만들고 `init()`으로 격자와 업데이트 계수를 준비한다.
5. 관측 대상을 등록한 뒤 `step_until_t()`로 시간을 전진시킨다.

가장 작은 예인 [`air2d.py`](../../examples/air2d.py)는 이 흐름을 그대로 보여 준다.

```python
from gmes import *

space = Cartesian(size=(10, 10, 0), resolution=20)
geometry = [
    DefaultMedium(material=Dielectric()),
    Shell(material=Cpml()),
]
sources = [
    PointSource(
        src_time=Continuous(freq=0.8),
        center=(0, 0, 0),
        component=Ez,
    )
]

simulation = TMzFDTD(space, geometry, sources)
simulation.init()
simulation.show_field(Ez, Z, 0)
simulation.step_until_t(10)
```

`size=(10, 10, 0)`은 z 방향 두께가 없는 2차원 영역을 뜻한다. `resolution=20`은 단위 길이마다 20개의 격자점을 사용한다는 의미다. 해상도를 높이면 작은 구조와 짧은 파장을 더 잘 표현하지만 메모리 사용량과 실행 시간이 빠르게 늘어난다.

배경은 `Dielectric()`로 채우고 바깥쪽에는 `Cpml` 셸을 둔다. CPML(Convolutional Perfectly Matched Layer)은 계산 영역 끝에서 파동이 인위적으로 되돌아오는 현상을 줄이는 흡수 경계다. 중앙의 `Ez` 점 소스는 주파수 0.8로 연속 진동하며, 2차원 TMz 문제에서는 `Ez`, `Hx`, `Hy` 성분이 함께 전파된다.

## 첫 관찰: 공기 중 원통파

`air2d.py`는 가장 먼저 실행하기 좋은 예제다. 소스에서 퍼져 나가는 `Ez`는 원형 파면을 만들고, `Hx`와 `Hy`는 서로 다른 공간 대칭성을 보인다. 복잡한 구조가 없기 때문에 다음 항목을 확인하는 기준 문제로도 유용하다.

- 설치와 네이티브 확장이 정상인지
- CPML 경계에서 눈에 띄는 반사가 생기지 않는지
- 격자 해상도에 따라 파면이 어떻게 달라지는지
- `show_field()`와 `write_field()`가 원하는 성분을 다루는지

최근 저장소 검증에서는 전체 `t=10` 실행이 Apple silicon 환경에서 약 3.2초 걸렸다. 장비와 빌드 설정에 따라 시간은 달라질 수 있지만, 다른 예제를 실행하기 전의 빠른 스모크 테스트로 적합하다.

## 유전체 슬래브 도파관

[`slab_waveguide.py`](../../examples/slab_waveguide.py)는 굴절률이 높은 유전체 코어에 빛을 가두는 가장 단순한 도파관을 모델링한다.

```python
space = Cartesian(size=(16, 8, 0), resolution=10)
geometry = [
    DefaultMedium(material=Dielectric()),
    Block(material=Dielectric(12), size=(inf, 1, inf)),
    Shell(material=Cpml()),
]
sources = [
    PointSource(
        src_time=Continuous(freq=0.15),
        component=Ez,
        center=(-7, 0, 0),
    )
]
```

상대 유전율 12인 폭 1의 블록이 코어다. 소스는 왼쪽에서 `Ez`를 여기하고, 진행하면서 에너지가 슬래브 부근에 모이는 모드를 관찰한다. 이 예제는 재료 하나만 추가해도 파동 전파가 얼마나 크게 바뀌는지 보여 준다.

다만 현재 GMES에는 `numpy.inf`를 형상 크기로 사용할 때 일관되게 무한대로 처리되지 않는 알려진 제한이 있다. 새 모델을 작성할 때는 계산 영역보다 충분히 큰 **유한한 값**으로 블록 크기를 지정하는 편이 안전하다.

## 광결정 선결함 도파관

[`phc_waveguide.py`](../../examples/phc_waveguide.py)는 규칙적으로 배열한 유전체 원기둥에서 한 줄을 비워 도파로를 만든다.

```python
geometry.extend(
    [
        Cylinder(material=Dielectric(8.9), radius=0.38, center=(x, y, 0))
        for x in range(-8, 9)
        for y in range(-4, 5)
        if y != 0
    ]
)
```

핵심은 `if y != 0`이다. 주기 구조의 가운데 행을 제거하면서 선결함이 생기고, 광결정 밴드갭 안의 빛이 이 통로를 따라 전달될 수 있다. 단순한 Python 리스트 컴프리헨션으로 반복 구조와 결함을 함께 표현했다는 점도 눈여겨볼 만하다.

이 예제에서는 `show_permittivity(Ez, Z, 0)`으로 먼저 재료 분포를 확인한 뒤 `show_field(Ez, Z, 0)`으로 장을 관찰한다. 복잡한 구조일수록 시뮬레이션 전에 유전율 단면을 확인하는 습관이 중요하다. 원기둥의 위치나 결함 행을 잘못 지정해도 코드는 정상 실행될 수 있기 때문이다.

## TFSF로 평면파와 산란 분리하기

점 소스 대신 평면파를 넣고 싶다면 [`tfsf.py`](../../examples/tfsf.py)의 `TotalFieldScatteredField` 소스를 참고할 수 있다.

```python
TotalFieldScatteredField(
    src_time=Continuous(freq=0.8),
    center=(0, 0, 0),
    size=(3, 3, 1),
    direction=(1, -1, 0),
    polarization=(0, 0, 1),
)
```

TFSF(total-field/scattered-field)는 전체장이 존재하는 영역과 산란장만 존재하는 영역을 나눈다. `direction`은 입사 방향, `polarization`은 편광 방향을 정한다. 이 예제에서는 z 편광 평면파가 대각선 방향으로 진행한다.

[`tfsf_with_scatterer.py`](../../examples/tfsf_with_scatterer.py)는 같은 설정의 중앙에 상대 유전율 3, 반지름 1인 원기둥을 추가한다.

```python
Cylinder(
    Dielectric(3),
    center=(0, 0, 0),
    radius=1,
    axis=(0, 0, 1),
)
```

두 예제를 나란히 실행하면 입사 평면파만 있을 때와 유전체 산란체가 있을 때의 차이를 분리해 볼 수 있다. 이후 원기둥의 반지름, 유전율, 입사각, 주파수를 바꿔 가며 산란 패턴을 비교하는 실험으로 확장하기 좋다.

## 분산 재료: 금 박막의 프레넬 반사

금속의 유전율은 일반 유전체처럼 상수 하나로 표현하기 어렵다. [`fresnel_reflection.py`](../../examples/fresnel_reflection.py)는 Drude pole과 두 개의 critical point를 조합한 `DcpPlrc` 모델로 금의 주파수 분산을 표현한다.

```python
dp = DrudePole(omega=..., gamma=...)
cp1 = CriticalPoint(amp=..., phi=..., omega=..., gamma=...)
cp2 = CriticalPoint(amp=..., phi=..., omega=..., gamma=...)
gold = DcpPlrc(
    eps_inf=1.11683,
    mu_inf=1,
    dps=(dp,),
    cps=(cp1, cp2),
)
```

소스는 `GaussianBeam`이고, 금 박막 앞뒤에 probe를 두어 반사파와 투과파의 시간 신호를 기록한다. `bloch=(0, k0 * sin(angle), 0)`은 경사 입사 문제의 횡방향 위상 변화를 나타낸다. 현재 예제의 `angle=0`을 바꾸면 입사각에 따른 반사와 투과를 조사할 수 있다.

이 예제를 다른 파장이나 금속으로 확장할 때는 단순히 `eps_inf`만 바꾸면 안 된다. 분산 모델의 계수, 단위 정규화, 적용 가능한 파장 범위를 함께 검토해야 물리적으로 의미 있는 결과를 얻는다.

## 3차원으로 확장하기

저장소에는 세 가지 성격이 다른 3차원 예제가 있다.

### 형상 API를 익히는 `man.py`

[`man.py`](../../examples/man.py)는 `Block`, `Sphere`, `Cone`, `Cylinder`, `Ellipsoid`를 조합해 사람 모양 구조를 만든다. 장을 전파시키기보다 형상 배치와 유전율 단면을 확인하는 예제다. 방향 벡터와 중심점, 크기가 3차원 객체에 어떻게 적용되는지 익히기 좋다.

빠르게 구조만 확인하려면 축소 옵션을 사용한다.

```sh
python examples/man.py --quick
```

### 평판형 광결정 `phc_slab.py`

[`phc_slab.py`](../../examples/phc_slab.py)는 실리콘-온-인슐레이터 기반의 평판 구조에 삼각 격자 공기 구멍과 선결함을 만든다. `make_crystals()`, `make_line_defect()`처럼 구조 생성을 함수로 분리했기 때문에 큰 반복 형상을 관리하는 방법을 보여 준다.

전체 모델은 역사적으로 약 1.3GB의 메모리를 요구한다. 먼저 축소 실행으로 설치와 형상 구성을 확인하는 편이 좋다.

```sh
python examples/phc_slab.py --quick
```

### 은 나노입자 배열 `metal_array.py`

[`metal_array.py`](../../examples/metal_array.py)는 여섯 개 은 나노구가 만드는 플라즈몬 도파관을 모델링한다. 은은 실험 데이터에 맞춘 분산 재료로 정의하고, 배열 방향과 같은 방향의 `Jy` 점 소스로 종방향 모드를 여기한다.

```python
for y in range(-2, 4):
    geometry.append(
        Sphere(Silver(75 * NANO), radius=1.0 / 3, center=(0, y, 0))
    )
```

전체 실행은 약 1.1GB의 메모리를 사용할 수 있으므로 먼저 다음 명령으로 축소 검증한다.

```sh
python examples/metal_array.py --quick
```

MPI 환경이 준비되어 있다면 병렬 옵션을 설치한 뒤 여러 프로세스로 실행할 수 있다.

```sh
python -m pip install -e ".[mpi]"
mpiexec -n 4 python examples/metal_array.py
```

## 어떤 예제부터 시작할까

| 목적 | 추천 예제 | 핵심 API | 실행 부담 |
| --- | --- | --- | --- |
| 설치와 기본 전파 확인 | `air2d.py` | `TMzFDTD`, `PointSource`, `Cpml` | 낮음 |
| 유전체 도파 모드 관찰 | `slab_waveguide.py` | `Block`, `Dielectric` | 낮음 |
| 주기 구조와 결함 학습 | `phc_waveguide.py` | `Cylinder`, 반복 형상 | 보통 |
| 평면파 입사 확인 | `tfsf.py` | `TotalFieldScatteredField` | 낮음 |
| 유전체 산란 비교 | `tfsf_with_scatterer.py` | TFSF, `Cylinder` | 낮음 |
| 반사율·투과율 측정 | `fresnel_reflection.py` | 분산 재료, `GaussianBeam`, probe | 보통 |
| 3차원 형상 구성 | `man.py --quick` | 3D 기하 객체 | 축소 시 낮음 |
| 3차원 광결정 | `phc_slab.py --quick` | 격자 생성, 선결함 | 전체 실행 높음 |
| 플라즈모닉 배열 | `metal_array.py --quick` | 분산 금속, MPI | 전체 실행 높음 |

처음이라면 `air2d.py`에서 해상도와 소스 주파수를 바꾸고, 다음으로 `slab_waveguide.py`에서 코어 폭과 유전율을 바꿔 보는 순서를 권한다. 그 뒤 `tfsf.py`와 산란체 버전을 비교하면 소스, 재료, 경계가 결과에 미치는 영향을 단계적으로 익힐 수 있다.

## 자신의 시뮬레이션으로 확장하는 방법

예제를 복사해 새 문제를 만들 때는 한 번에 하나의 요소만 바꾸는 것이 좋다.

1. 계산 영역과 구조의 물리적 크기를 정한다.
2. 관심 있는 최소 파장을 기준으로 해상도를 선택한다.
3. 배경과 구조물의 재료를 배치하고 유전율 단면을 확인한다.
4. 소스의 위치, 성분, 주파수, 편광을 정한다.
5. CPML 두께와 구조물 사이에 충분한 간격을 둔다.
6. 짧은 시간과 낮은 해상도로 먼저 실행한다.
7. probe나 필드 출력으로 원하는 물리량이 실제로 측정되는지 확인한다.
8. 마지막에 해상도와 실행 시간을 늘리고 격자 수렴성을 확인한다.

FDTD 결과는 그림이 자연스러워 보인다는 이유만으로 신뢰할 수 없다. 해상도를 높였을 때 결과가 수렴하는지, 경계 반사가 충분히 작은지, 에너지 보존과 알려진 해석해를 만족하는지 확인해야 한다. 금속이나 손실 재료를 사용한다면 시간 간격과 재료 모델의 안정성도 별도로 살펴야 한다.

## 마치며

GMES 예제의 장점은 복잡한 전자기 문제도 결국 동일한 조립 과정으로 표현된다는 점이다. `Cartesian`으로 무대를 만들고, 기하 객체로 재료를 배치하고, 소스를 넣고, 적절한 FDTD 클래스로 시간을 전진시킨다. 공기 중 원통파와 슬래브 도파관에서 이 패턴을 익히면 광결정, TFSF 산란, 분산 금속 같은 고급 예제도 훨씬 쉽게 읽힌다.

다음 단계는 예제 하나를 기준 문제로 삼아 매개변수를 체계적으로 바꾸고, probe와 필드 출력을 이용해 정량적인 결과를 얻는 것이다. 저장소의 [`examples/VERIFICATION.md`](../../examples/VERIFICATION.md)에는 각 예제의 최근 검증 범위와 실행 비용이 정리되어 있으므로 큰 계산을 시작하기 전에 함께 확인할 수 있다.
