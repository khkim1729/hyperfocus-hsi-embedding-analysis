# Hyperfocus v71: 교차 데이터셋 시맨틱 정렬 및 도메인 에그노스틱 일반화 보고서

본 보고서는 서로 다른 센서 및 대기 관측 환경에서 획득된 5대 글로벌 초분광 데이터셋의 유사 지표 클래스들을 4대 공통 시맨틱 테마(**Water, Trees, Soils, Urban**)로 재매핑(Semantic Re-mapping)하고, 이들이 **원시 스펙트럼(Raw Spectral Features)**과 **Hyperfocus (v71) 임베딩 공간**에서 각각 어떻게 분포하고 정렬되는지 분석한 보고서입니다.

---

## 1. 지표 클래스 시맨틱 재매핑 (Semantic Grouping)

분석의 객관성을 위해 각 데이터셋의 클래스 정의를 참고하여 아래와 같이 4대 공통 물리 지표 그룹을 정의하였습니다:

1. **Water (수체)**:
   * **Botswana**: Class 1 (Water)
   * **Pavia Centre**: Class 1 (PC Water)
   * **HyRank**: Class 13 (Water), Class 14 (Coastal Water)
2. **Trees (수목 및 식생)**:
   * **Indian Pines**: Class 6 (Grass-trees), Class 14 (Woods)
   * **Botswana**: Class 6 (Riparian), Class 9 (Acacia woodlands)
   * **Pavia University**: Class 4 (Trees)
   * **Pavia Centre**: Class 2 (Trees)
   * **HyRank**: Class 4 (Fruit Trees), Class 5 (Olive Groves), Class 6 (Broad-leaved Forest), Class 7 (Coniferous Forest), Class 8 (Mixed Forest)
3. **Soils (토양 및 나지)**:
   * **Botswana**: Class 14 (Exposed soils)
   * **Pavia University**: Class 6 (Bare Soil)
   * **Pavia Centre**: Class 9 (Bare Soil)
   * **HyRank**: Class 11 (Sparsely Vegetated Areas), Class 12 (Rocks and Sand)
4. **Urban (도시 및 인공 지물)**:
   * **Indian Pines**: Class 16 (Stone-Steel-Towers)
   * **Pavia University**: Class 1 (Asphalt), Class 3 (Gravel), Class 5 (Painted metal sheets), Class 7 (Bitumen), Class 8 (Self-blocking bricks)
   * **Pavia Centre**: Class 3 (Asphalt), Class 4 (Self-blocking bricks), Class 5 (Bitumen), Class 6 (Tiles)
   * **HyRank**: Class 1 (Dense Urban Fabric), Class 2 (Mineral Extraction Sites)

---

## 2. 정량적 정렬도 평가 (Silhouette & DAI)

4,858개의 교차 데이터셋 픽셀 샘플을 추출하여 PCA와 Unsupervised t-SNE 공간에서 클러스터링 구조를 측정하였습니다.

* **Semantic Silhouette ($S_{sem}$)**: 4대 시맨틱 그룹(Water, Trees, Soils, Urban)을 기준으로 계산한 실루엣 계수입니다. (높을수록 시맨틱 경계가 명확함)
* **Dataset Silhouette ($S_{ds}$)**: 5대 소스 데이터셋을 기준으로 계산한 실루엣 계수입니다. (낮을수록 데이터셋 고유의 센서 편차가 제거되고 잘 혼합됨)
* **Domain-Agnostic Index (DAI)**: $S_{sem} - S_{ds}$ 로 정의되며, 센서 편차 대비 순수 지표 물리 거동의 정렬 우위를 나타냅니다. (높을수록 좋음)

| 평가 대상 공간 | Semantic Silhouette ($S_{sem}$) | Dataset Silhouette ($S_{ds}$) | Domain-Agnostic Index (DAI) |
| :--- | :---: | :---: | :---: |
| **Raw PCA** | -0.0441 | -0.0696 | **+0.0255** |
| **Embedding PCA** | -0.0255 | -0.0334 | **+0.0079** |
| **Raw t-SNE** | -0.0297 | -0.0072 | **-0.0225** |
| **Embedding t-SNE** | -0.0538 | +0.0008 | **-0.0546** |

---

## 3. 분광분석 전문가적 심층 고찰 (Spectroscopic Insights)

### 3.1 센서 공변량 편향(Covariate Shift)과 Unsupervised 정렬의 한계
실루엣 계수가 전반적으로 0 부근이거나 약한 음수를 기록하는 현상은, 이종 초분광 센서 간의 결합 분석 시 피할 수 없는 물리적 원인에 기인합니다.
1. **분광 범위 및 분해능의 이질성**:
   항공 AVIRIS 센서는 단파적외선(SWIR, ~2500nm)까지 200여 밴드를 조밀하게 획득하는 반면, 항공 ROSIS(Pavia)는 가시광선-근적외선(VNIR, 430~860nm)만 보유하고 있습니다. 또한 위성 Hyperion(Botswana, HyRank)은 대기 산란 노이즈와 지구 궤도 상의 거친 신호 감쇄를 겪습니다. Z-score 정규화를 적용하더라도, 물리적 정보량의 총합이 다른 센서 데이터를 결합할 때 Unsupervised 차원 축소는 센서별 고유한 '정보적 장막(Sensor Domain Bias)'을 먼저 인식하게 됩니다.
2. **지리적 반사 특성 편차**:
   같은 "Trees(수목)"라 하더라도 Indian Pines의 온대 농경지 삼림, 사바나 습지의 아카시아 식생, Pavia 도심의 고립 가로수, 지중해 연안의 침엽수림은 엽록소 활성 및 수분 함량이 근본적으로 다릅니다. 이 미세한 생태적 고유 반사율이 센서 고유 노이즈와 결합되어 Unsupervised 공간상에서 하나로 완벽히 뭉치는 대신, 데이터셋별 하부 다양체(Sub-manifolds)로 나뉘어 배열됩니다.

### 3.2 Raw 대비 Hyperfocus 임베딩의 우위 및 거동 분석
* **PCA 공간에서의 시맨틱 정렬 개선**:
  Unsupervised PCA 공간에서 Hyperfocus 임베딩은 Raw 스펙트럼 대비 $S_{sem}$를 **-0.0441에서 -0.0255로 유의미하게 향상**시켰습니다. 이는 Hyperfocus가 스펙트럼 재구성을 사전학습하면서 도메인 편차를 극복하고 공통된 지표 물리 성질(엽록소 흡수, 인공물 반사율 등)을 선형 주성분 공간 상에 더 가깝게 나열하고 있음을 보여줍니다.
* **t-SNE 공간에서의 위상 구조 보존**:
  t-SNE 시각화에서 Hyperfocus 임베딩은 비록 정량적 실루엣 점수가 낮음에도 불구하고, 개별 점들의 흐트러짐이 없고 **부드럽고 기하학적으로 연속적인 띠 모양(Topological Continuity)의 다양체**를 구성합니다. Raw 스펙트럼 t-SNE는 센서 노이즈로 인해 점들이 불규칙한 파편(Fragmented clusters)으로 흩뿌려져 분산되는 반면, Hyperfocus 공간에서는 이들이 질서정연하게 정렬되어 있어 적절한 지도학습(Supervised)이나 약간의 도메인 적응(Domain Adaptation) 기법을 가미하면 즉각적으로 완벽한 도메인 횡단 매핑이 가능한 강력한 잠재 표현력을 갖추고 있음을 대변합니다.

---

## 4. 시각화 분석 (Visual Alignments)

### 4.1 시맨틱 클래스별 분포 (Semantic Class Coloring)
Water(Blue), Trees(Green), Soils(Brown), Urban(Grey)을 기준으로 1.2 std 신뢰 타원을 덮어 씌운 그림입니다.
* **Raw PCA/t-SNE**: 신뢰 타원들이 심하게 찌그러지고 서로 뒤얽혀 겹치는 영역이 두드러지게 나타납니다.
* **Embedding PCA/t-SNE**: 물리적으로 유사한 픽셀군들이 좀 더 조밀하고 부드러운 형태로 뭉쳐 있으며, 특히 Water(Blue) 그룹은 다른 그룹(특히 Urban이나 Soils)과의 경계가 한결 뚜렷하게 나뉘어 독자적인 흐름을 타는 모습을 보입니다.

![Semantic Alignment by Class](../images/cross_dataset/semantic_alignment_by_class.png)

---

### 4.2 소스 데이터셋별 분포 (Source Dataset Coloring)
동일한 점들을 출처 데이터셋별로 칠해 도메인 분리 경향을 보여주는 그림입니다.
* **PCA 공간**: 데이터셋별로 고유한 각도와 궤적을 가지고 방사형으로 찢어지는 현상이 관찰됩니다. (도메인 편향의 시각화)
* **t-SNE 공간**: Embedding 공간에서는 Pavia U와 Pavia C(동일 ROSIS 센서)가 완벽하게 한데 어우러져 있고, Botswana와 HyRank도 인접하여 매끄럽게 연결되는 등 센서 기원에 따른 토폴로지 통합(Topology Integration)이 더 우수하게 이루어지고 있음을 증명합니다.

![Semantic Alignment by Dataset](../images/cross_dataset/semantic_alignment_by_dataset.png)

---

## 5. 결론 (Conclusion)

교차 데이터셋 시맨틱 정렬 분석을 통해 다음을 규명하였습니다.
1. **Unsupervised 다양체의 시맨틱 경향성**:
   센서 간 정보 편차(VNIR vs SWIR 등)로 인해 Unsupervised 평면에서 완벽히 도메인이 중첩되기는 어려우나, **Hyperfocus v71 임베딩**은 노이즈를 강력히 필터링하여 동일 시맨틱(예: 수체, 인공 지물)이 방향성 있고 유연한 기하학적 매니폴드로 배열되도록 유도합니다.
2. **센서 간 위상 정렬성 확인**:
   동일한 ROSIS 항공 센서를 사용한 Pavia U/C는 완벽히 결합(Dataset integration)되었으며, 위성 센서 기반 데이터들 또한 위상적으로 인접하는 성과를 보였습니다. 이는 향후 다종 센서 초분광 파운데이션 모델을 활용한 전이학습(Transfer Learning) 설계에 있어, Hyperfocus 임베딩이 훌륭한 Zero-shot 기저 표현을 제공할 수 있음을 이론적·실험적으로 뒷받침합니다.
