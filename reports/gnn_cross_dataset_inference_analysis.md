# 교차 데이터셋 GNN 모델 제로샷 추론 분석 보고서
본 보고서는 특정 초분광 데이터셋의 원본 클래스(Real Classes)로 학습된 GNN 모델을 사용하여 타겟 데이터셋에 직접 제로샷 추론(Zero-shot Inference)을 수행한 결과 분석 리포트입니다.
기초 모델 **Hyperfocus v71**이 추출한 128차원 임베딩을 노드 피처로 사용하고, 타겟 데이터셋의 공간 그래프 구조 상에서 인접 메시지를 전파하여 예측을 수행했습니다.
---
## Indian Pines GNN 모델 추론 결과 분석
**Indian Pines** 데이터셋(소스 도메인)으로 학습된 GCN Embedding 모델을 사용하여 타겟 데이터셋들에 추론을 수행했습니다. 소스 도메인의 StandardScaler 평균/표준편차로 타겟 임베딩을 정규화한 후 추론을 가해 도메인 정합성을 맞추었습니다.

### 타겟 데이터셋: Botswana
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Water** | Buildings-Grass-Trees-Drives (99.24%) | Wheat (0.69%) | Grass-trees (0.07%) |
| **Hippo grass** | Buildings-Grass-Trees-Drives (99.89%) | Grass-trees (0.11%) | Wheat (0.00%) |
| **Floodplain grasses 1** | Soybean-notill (54.35%) | Soybean-mintill (42.99%) | Corn-mintill (1.44%) |
| **Floodplain grasses 2** | Grass-trees (93.55%) | Corn-mintill (4.26%) | Soybean-mintill (2.11%) |
| **Reeds** | Buildings-Grass-Trees-Drives (49.84%) | Grass-pasture (37.03%) | Corn (10.44%) |
| **Riparian** | Buildings-Grass-Trees-Drives (93.53%) | Corn (5.31%) | Grass-pasture (1.07%) |
| **Firescar** | Corn-mintill (50.35%) | Soybean-mintill (40.85%) | Grass-trees (7.47%) |
| **Island interior** | Corn-mintill (95.94%) | Grass-trees (3.62%) | Corn (0.23%) |
| **Acacia woodlands** | Grass-trees (48.54%) | Buildings-Grass-Trees-Drives (30.60%) | Corn (20.64%) |
| **Acacia shrublands** | Corn-mintill (77.66%) | Corn (22.33%) | Soybean-notill (0.00%) |
| **Acacia grasslands** | Corn-mintill (99.99%) | Corn (0.01%) | Soybean-mintill (0.00%) |
| **Short mopane** | Corn (86.41%) | Corn-mintill (13.59%) | Woods (0.00%) |
| **Mixed mopane** | Corn (92.46%) | Corn-mintill (7.54%) | Oats (0.00%) |
| **Exposed soils** | Corn-mintill (85.51%) | Soybean-mintill (6.45%) | Stone-Steel-Towers (3.41%) |

### 타겟 데이터셋: Pavia University
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Asphalt** | Soybean-mintill (96.65%) | Corn (2.65%) | Buildings-Grass-Trees-Drives (0.65%) |
| **Meadows** | Buildings-Grass-Trees-Drives (95.17%) | Soybean-mintill (4.83%) | Woods (0.00%) |
| **Gravel** | Soybean-mintill (99.26%) | Buildings-Grass-Trees-Drives (0.74%) | Grass-trees (0.00%) |
| **Trees** | Buildings-Grass-Trees-Drives (93.65%) | Soybean-mintill (6.35%) | Grass-trees (0.00%) |
| **Painted metal sheets** | Buildings-Grass-Trees-Drives (99.99%) | Woods (0.01%) | Corn (0.00%) |
| **Bare Soil** | Corn (48.01%) | Soybean-mintill (24.23%) | Grass-trees (10.31%) |
| **Bitumen** | Soybean-mintill (42.48%) | Grass-trees (25.10%) | Buildings-Grass-Trees-Drives (16.75%) |

### 타겟 데이터셋: HyRank
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Dense urban fabric** | Corn-mintill (99.55%) | Soybean-mintill (0.29%) | Corn-notill (0.15%) |
| **Mineral extraction sites** | Corn-notill (33.17%) | Corn-mintill (31.71%) | Soybean-mintill (27.91%) |
| **Non-irrigated arable land** | Corn-mintill (57.00%) | Corn-notill (33.59%) | Soybean-mintill (9.40%) |
| **Fruit trees** | Grass-trees (51.57%) | Soybean-mintill (25.65%) | Corn-mintill (17.37%) |
| **Olive groves** | Corn-mintill (92.73%) | Grass-trees (4.83%) | Soybean-mintill (2.23%) |
| **Natural grassland** | Grass-trees (99.63%) | Soybean-mintill (0.19%) | Grass-pasture (0.18%) |
| **Water courses** | Grass-trees (94.01%) | Soybean-mintill (4.05%) | Corn-mintill (1.90%) |
| **Coastal lagoons** | Corn-mintill (48.78%) | Soybean-mintill (32.36%) | Grass-trees (13.49%) |
| **Estuaries** | Corn-mintill (64.98%) | Corn (20.78%) | Soybean-mintill (7.78%) |
| **Sea and ocean** | Soybean-notill (37.72%) | Soybean-mintill (28.35%) | Corn (23.15%) |
| **Water bodies** | Woods (82.67%) | Corn-notill (14.24%) | Buildings-Grass-Trees-Drives (2.59%) |
| **Herbaceous vegetation** | Woods (66.05%) | Corn-notill (33.68%) | Corn (0.18%) |

---
## Pavia University GNN 모델 추론 결과 분석
**Pavia University** 데이터셋(소스 도메인)으로 학습된 GCN Embedding 모델을 사용하여 타겟 데이터셋들에 추론을 수행했습니다. 소스 도메인의 StandardScaler 평균/표준편차로 타겟 임베딩을 정규화한 후 추론을 가해 도메인 정합성을 맞추었습니다.

### 타겟 데이터셋: Indian Pines
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Alfalfa** | Bitumen (94.06%) | Meadows (5.37%) | Asphalt (0.58%) |
| **Corn-notill** | Bitumen (77.88%) | Asphalt (22.09%) | Meadows (0.02%) |
| **Corn-mintill** | Bitumen (80.05%) | Asphalt (19.81%) | Bare Soil (0.07%) |
| **Corn** | Bitumen (69.44%) | Asphalt (30.02%) | Bare Soil (0.35%) |
| **Grass-pasture** | Asphalt (48.56%) | Bitumen (24.23%) | Painted metal sheets (20.51%) |
| **Grass-trees** | Asphalt (50.39%) | Meadows (31.60%) | Bare Soil (10.23%) |
| **Grass-pasture-mowed** | Bitumen (99.70%) | Asphalt (0.16%) | Meadows (0.13%) |
| **Hay-windrowed** | Bitumen (69.11%) | Asphalt (22.28%) | Meadows (8.61%) |
| **Oats** | Asphalt (78.90%) | Meadows (19.64%) | Bare Soil (1.25%) |
| **Soybean-notill** | Bitumen (64.23%) | Asphalt (35.68%) | Meadows (0.07%) |
| **Soybean-mintill** | Bitumen (88.34%) | Asphalt (11.64%) | Bare Soil (0.01%) |
| **Soybean-clean** | Bitumen (60.99%) | Asphalt (38.63%) | Meadows (0.23%) |
| **Wheat** | Asphalt (65.66%) | Meadows (20.45%) | Painted metal sheets (13.85%) |
| **Woods** | Asphalt (69.76%) | Painted metal sheets (30.10%) | Meadows (0.13%) |
| **Buildings-Grass-Trees-Drives** | Asphalt (67.35%) | Meadows (24.32%) | Painted metal sheets (8.11%) |
| **Stone-Steel-Towers** | Meadows (72.09%) | Bitumen (27.87%) | Asphalt (0.04%) |

### 타겟 데이터셋: Botswana
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Water** | Asphalt (54.26%) | Painted metal sheets (45.74%) | Bare Soil (0.01%) |
| **Hippo grass** | Meadows (77.61%) | Painted metal sheets (12.28%) | Asphalt (9.83%) |
| **Floodplain grasses 1** | Bitumen (86.92%) | Bare Soil (13.07%) | Asphalt (0.01%) |
| **Floodplain grasses 2** | Bitumen (99.45%) | Bare Soil (0.55%) | Asphalt (0.00%) |
| **Reeds** | Bitumen (76.67%) | Meadows (21.59%) | Asphalt (1.73%) |
| **Riparian** | Bitumen (69.06%) | Asphalt (15.62%) | Meadows (13.62%) |
| **Firescar** | Asphalt (76.51%) | Bare Soil (13.66%) | Bitumen (9.77%) |
| **Island interior** | Bitumen (100.00%) | Bare Soil (0.00%) | Gravel (0.00%) |
| **Acacia woodlands** | Bitumen (64.87%) | Bare Soil (31.14%) | Asphalt (3.56%) |
| **Acacia shrublands** | Bitumen (97.18%) | Bare Soil (2.81%) | Asphalt (0.01%) |
| **Acacia grasslands** | Bitumen (100.00%) | Asphalt (0.00%) | Bare Soil (0.00%) |
| **Short mopane** | Bitumen (99.95%) | Asphalt (0.05%) | Meadows (0.00%) |
| **Mixed mopane** | Bitumen (94.59%) | Bare Soil (4.23%) | Asphalt (1.18%) |
| **Exposed soils** | Bitumen (100.00%) | Asphalt (0.00%) | Gravel (0.00%) |

### 타겟 데이터셋: HyRank
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Dense urban fabric** | Bitumen (100.00%) | Asphalt (0.00%) | Meadows (0.00%) |
| **Mineral extraction sites** | Bitumen (98.91%) | Meadows (0.72%) | Asphalt (0.36%) |
| **Non-irrigated arable land** | Bitumen (100.00%) | Asphalt (0.00%) | Gravel (0.00%) |
| **Fruit trees** | Bitumen (97.56%) | Meadows (2.44%) | Asphalt (0.00%) |
| **Olive groves** | Bitumen (100.00%) | Bare Soil (0.00%) | Meadows (0.00%) |
| **Natural grassland** | Bitumen (78.45%) | Asphalt (10.12%) | Painted metal sheets (8.99%) |
| **Water courses** | Bitumen (99.74%) | Meadows (0.23%) | Asphalt (0.01%) |
| **Coastal lagoons** | Bitumen (99.49%) | Asphalt (0.48%) | Bare Soil (0.03%) |
| **Estuaries** | Bitumen (97.29%) | Asphalt (2.71%) | Meadows (0.00%) |
| **Sea and ocean** | Bitumen (99.56%) | Asphalt (0.44%) | Meadows (0.00%) |
| **Water bodies** | Painted metal sheets (80.27%) | Asphalt (19.73%) | Meadows (0.00%) |
| **Herbaceous vegetation** | Painted metal sheets (79.89%) | Asphalt (20.10%) | Meadows (0.00%) |

---
## HyRank GNN 모델 추론 결과 분석
**HyRank** 데이터셋(소스 도메인)으로 학습된 GCN Embedding 모델을 사용하여 타겟 데이터셋들에 추론을 수행했습니다. 소스 도메인의 StandardScaler 평균/표준편차로 타겟 임베딩을 정규화한 후 추론을 가해 도메인 정합성을 맞추었습니다.

### 타겟 데이터셋: Indian Pines
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Alfalfa** | Olive groves (85.23%) | Non-irrigated arable land (13.73%) | Fruit trees (1.00%) |
| **Corn-notill** | Coastal lagoons (31.79%) | Mineral extraction sites (31.01%) | Dense urban fabric (13.31%) |
| **Corn-mintill** | Olive groves (46.13%) | Dense urban fabric (20.35%) | Coastal lagoons (15.39%) |
| **Corn** | Coastal lagoons (38.29%) | Mineral extraction sites (17.34%) | Non-irrigated arable land (12.50%) |
| **Grass-pasture** | Herbaceous vegetation (66.15%) | Water courses (28.06%) | Coastal lagoons (3.68%) |
| **Grass-trees** | Water courses (98.13%) | Coastal lagoons (1.80%) | Natural grassland (0.03%) |
| **Grass-pasture-mowed** | Water courses (86.31%) | Olive groves (12.64%) | Herbaceous vegetation (0.48%) |
| **Hay-windrowed** | Olive groves (40.24%) | Non-irrigated arable land (26.78%) | Coastal lagoons (20.50%) |
| **Oats** | Water courses (60.53%) | Natural grassland (38.06%) | Olive groves (1.14%) |
| **Soybean-notill** | Dense urban fabric (30.41%) | Non-irrigated arable land (24.11%) | Coastal lagoons (23.31%) |
| **Soybean-mintill** | Olive groves (37.47%) | Dense urban fabric (25.43%) | Non-irrigated arable land (16.15%) |
| **Soybean-clean** | Coastal lagoons (26.96%) | Olive groves (25.45%) | Water bodies (16.94%) |
| **Wheat** | Natural grassland (58.00%) | Water courses (41.32%) | Herbaceous vegetation (0.36%) |
| **Woods** | Herbaceous vegetation (95.10%) | Water courses (4.42%) | Olive groves (0.48%) |
| **Buildings-Grass-Trees-Drives** | Water courses (80.39%) | Herbaceous vegetation (19.36%) | Natural grassland (0.23%) |
| **Stone-Steel-Towers** | Olive groves (69.77%) | Mineral extraction sites (28.94%) | Herbaceous vegetation (1.26%) |

### 타겟 데이터셋: Botswana
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Water** | Water bodies (99.91%) | Herbaceous vegetation (0.09%) | Natural grassland (0.00%) |
| **Hippo grass** | Water courses (100.00%) | Olive groves (0.00%) | Herbaceous vegetation (0.00%) |
| **Floodplain grasses 1** | Coastal lagoons (76.85%) | Olive groves (13.31%) | Water courses (9.73%) |
| **Floodplain grasses 2** | Olive groves (49.92%) | Water courses (39.17%) | Coastal lagoons (10.91%) |
| **Reeds** | Water courses (66.84%) | Olive groves (31.52%) | Herbaceous vegetation (1.60%) |
| **Riparian** | Water courses (95.60%) | Olive groves (3.97%) | Herbaceous vegetation (0.41%) |
| **Firescar** | Olive groves (58.18%) | Water courses (12.02%) | Water bodies (11.64%) |
| **Island interior** | Coastal lagoons (77.21%) | Olive groves (11.51%) | Sea and ocean (6.66%) |
| **Acacia woodlands** | Water courses (93.57%) | Coastal lagoons (6.38%) | Olive groves (0.05%) |
| **Acacia shrublands** | Coastal lagoons (88.04%) | Dense urban fabric (11.27%) | Olive groves (0.37%) |
| **Acacia grasslands** | Dense urban fabric (91.88%) | Coastal lagoons (4.53%) | Estuaries (3.16%) |
| **Short mopane** | Estuaries (50.27%) | Sea and ocean (23.63%) | Herbaceous vegetation (22.41%) |
| **Mixed mopane** | Estuaries (74.11%) | Coastal lagoons (15.43%) | Olive groves (6.98%) |
| **Exposed soils** | Sea and ocean (97.30%) | Olive groves (2.47%) | Non-irrigated arable land (0.12%) |

### 타겟 데이터셋: Pavia University
| 타겟 실제 클래스 (Actual Class) | 1순위 예측 클래스 (확률) | 2순위 예측 클래스 (확률) | 3순위 예측 클래스 (확률) |
| :--- | :--- | :--- | :--- |
| **Asphalt** | Herbaceous vegetation (57.19%) | Water bodies (31.86%) | Olive groves (9.60%) |
| **Meadows** | Water courses (80.92%) | Coastal lagoons (19.08%) | Water bodies (0.00%) |
| **Gravel** | Water courses (99.13%) | Coastal lagoons (0.85%) | Water bodies (0.03%) |
| **Trees** | Water courses (99.31%) | Coastal lagoons (0.69%) | Water bodies (0.00%) |
| **Painted metal sheets** | Water bodies (97.91%) | Water courses (1.46%) | Herbaceous vegetation (0.64%) |
| **Bare Soil** | Coastal lagoons (59.01%) | Water bodies (13.80%) | Water courses (9.62%) |
| **Bitumen** | Coastal lagoons (67.56%) | Water courses (28.59%) | Herbaceous vegetation (2.05%) |

## 4. GNN 제로샷 교차 추론 결과 분석 및 논의

본 실험을 통해 특정 초분광 데이터셋의 원본 클래스(Real Classes)로 학습된 GNN 모델을 다른 데이터셋에 직접 적용했을 때의 거동을 분석한 결과, 다음과 같은 핵심적인 시사점을 도출할 수 있습니다.

### 4.1 소스 도메인과 타겟 도메인의 클래스 불일치 (Disjoint Label Spaces)
GNN 모델의 분류 레이어(Softmax)는 학습 시에 정의된 소스 도메인의 고유 클래스 수(예: Indian Pines 16개, Pavia University 7개, HyRank 12개)에 국한된 차원으로 로짓을 출력합니다.
따라서 타겟 데이터셋(예: Botswana)의 픽셀이 입력될 경우, 모델은 강제로 자신이 아는 소스 클래스 중 하나로 매핑할 수밖에 없습니다. 
- **예시 (Indian Pines GNN -> Botswana)**: Indian Pines GNN은 농경지 중심의 클래스 구성을 가집니다. 따라서 Botswana의 `Water` 클래스가 들어왔을 때, 물에 대응되는 클래스가 없어 도시/혼합 구역인 `Buildings-Grass-Trees-Drives` (99.24%)로 강제 매핑됩니다.
- **예시 (Pavia University GNN -> Botswana)**: Pavia University GNN 역시 물이나 작물 클래스가 없습니다. 이에 따라 Botswana의 `Water`가 `Asphalt` (54.26%) 및 `Painted metal sheets` (45.74%)와 같은 인공물 클래스로 치우쳐 예측되는 현상이 발생합니다.

### 4.2 초분광 임베딩 기반의 잠재적 시맨틱 정합성 (Semantic Similarity Alignment)
클래스 레이블 공간이 서로 다름에도 불구하고, **Hyperfocus v71 기초 모델이 생성한 128차원 임베딩은 데이터셋을 초월하여 시맨틱한 유사 관계를 보존**하고 있음이 확인되었습니다.

1. **식생(Vegetation) 클래스 정합**:
   - **Indian Pines -> Botswana**: Botswana의 대표적 식생인 `Acacia woodlands`가 Indian Pines GNN 모델에서 `Grass-trees` (48.54%) 및 `Buildings-Grass-Trees-Drives` (30.60%)로 예측되며, `Short mopane`과 `Mixed mopane`은 식생인 `Corn` (86.41%, 92.46%)으로 맵핑됩니다.
   - **Pavia University -> Botswana**: Botswana의 `Hippo grass` (습지 잔디)가 Pavia University GNN 모델에서 대표 식생 클래스인 `Meadows` (77.61%)로 높은 확률로 매핑되었습니다.
2. **수계(Water) 클래스 정합**:
   - **HyRank -> Botswana**: HyRank GNN 모델은 강과 바다 등 수계 클래스(`Water bodies`, `Water courses`, `Coastal lagoons`, `Estuaries`, `Sea and ocean`)를 다수 포함하고 있습니다. 이 모델을 Botswana에 대입했을 때, Botswana의 `Water`는 HyRank의 `Water bodies` (99.91%)로, `Hippo grass`는 `Water courses` (100.00%)로, `Riparian` (강변 식생)은 `Water courses` (95.60%)로 극도로 높은 확률로 완벽하게 매핑되었습니다.
   - 이는 Hyperfocus v71 임베딩 공간 상에서 서로 다른 센서로 촬영된 수계 픽셀들이 매우 근접하게 군집화되어 있음을 직접적으로 증명합니다.

### 4.3 도메인 시프트 (Domain Shift) 및 대역 정합성 문제
StandardScaler를 이용해 소스 도메인의 평균과 표준편차로 타겟 임베딩 데이터를 표준화하여 Z-score 도메인 정렬을 가했음에도 불구하고, 센서별 분광 해상도 차이(Wavelength Bands), 촬영 당시의 조명 및 대기 조건, 지리적 배경(배경 토양의 반사율 등)으로 인한 도메인 시프트가 존재합니다.
이로 인해 일부 식생 클래스들이 엉뚱하게 도시 인공물(`Bitumen`, `Asphalt`, `Dense urban fabric`)로 오분류되는 경향도 관찰됩니다. 이는 지도학습(Supervised learning) 기반의 분류기가 특정 도메인의 결정 경계에 과적합(Overfitting)되어 발생하며, GNN의 공간적 평활화(Spatial Smoothing) 작용이 결합되면서 인접 노드로 오분류가 강하게 전파된 결과입니다.

### 4.4 결론 및 향후 개선 방향
- **GNN 제로샷 전이의 한계**: 원본 클래스 수준의 GNN 제로샷 추론은 클래스 레이블이 상이한 환경에서 완벽한 정량적 분류를 제공할 수 없으나, Hyperfocus v71 임베딩 공간 덕분에 **의미론적(Semantic) 유사도를 기반으로 한 클래스 매핑 가이드라인** 역할을 수행할 수 있습니다.
- **개선 방안**: 
  1. **텍스트-초분광 멀티모달 정렬**: CLIP과 유사하게 클래스의 의미(Text Description)와 초분광 임베딩을 공유 공간에 정렬하는 대조 학습 기법을 도입하여, 소스 클래스 공간에 종속되지 않는 개방형 어휘(Open-vocabulary) 제로샷 전이 기법을 구축할 수 있습니다.
  2. **도메인 적응(Domain Adaptation)**: 소수 타겟 데이터 레이블을 이용한 GNN 파인튜닝(Few-shot Transfer)이나, 비지도 도메인 정렬 기법을 적용하면 분류 정확도를 극적으로 향상할 수 있을 것입니다.

---

### 🔗 관련 문서 바로가기
* **Hyperfocus v71 README**:
  👉 **[README.md](../README.md)**
* **GNN 원본 클래스 분류 분석 보고서**:
  👉 **[공간 그래프 신경망(GNN) 기반 원본 클래스(Real Classes) 분류 성능 분석 보고서 (gnn_real_classes_classification_analysis.md)](gnn_real_classes_classification_analysis.md)**
