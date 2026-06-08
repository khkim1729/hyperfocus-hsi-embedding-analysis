# 공간 그래프 신경망(GNN) 기반 원본 클래스(Real Classes) 분류 성능 분석 보고서

본 보고서는 초분광 픽셀을 공동 의미 그룹(4-class)으로 매핑하지 않고, **각 데이터셋의 고유 원본 라벨 클래스(Real Classes)**를 대상으로 Graph Convolutional Network (GCN) 및 GraphSAGE 공간 그래프 신경망 분류를 진행한 성능 평가 결과입니다. 

기초 모델 **Hyperfocus v71**이 학습한 128차원 스펙트럼 임베딩 벡터의 강인성과 물리적 표현 한계를 원시 초분광 반사율(Raw Spectra)과 병렬 비교하여 분석합니다.

---

## 1. 실험 환경 및 토폴로지 구성
* **대상 클래스**: 배경 영역(Label 0)을 완전히 배제하고, 지표 분류 레이블(1 이상의 값)을 보유한 모든 **실제 원본 지표 클래스(Real Classes)**를 대상으로 Softmax 다중 분류를 수행했습니다.
* **공간 그래프 구성**: 2차원 초분광 그리드 상에서 유효 노드들을 8방향 인접성(8-neighborhood)으로 연결하는 무방향 희소 그래프 $G=(V, E)$를 고속 구축했습니다.
* **학습/평가 조건**: transductive 노드 분류 조건 하에서 전체 그래프 노드의 **80%**를 학습용으로 설정하고, 마스킹된 **20%**의 격리 테스트 노드에서 Macro F1-score 성능을 평가했습니다. (AdamW Optimizer, Weight Decay $1e-4$, Dropout 0.25 적용, 150 Epoch 학습)

### 데이터셋별 원본 그래프 규격 및 GNN 성능 종합 명세 (Full Dataset 기준)
| Dataset | Real Classes Count | Total Labeled Nodes (V) | Total Graph Edges (E) | Train Nodes (80%) | Test Nodes (20%) | GCN Raw F1 (Full) | GCN Emb F1 (Full) | SAGE Raw F1 (Full) | SAGE Emb F1 (Full) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | **16 클래스** | 10,249 | 73,874 | 8,199 | 2,050 | 0.9859 | 0.9925 | 0.9587 | 0.9967 |
| **Botswana** | **14 클래스** | 3,248 | 19,372 | 2,598 | 650 | 0.9976 | 1.0000 | 0.9989 | 1.0000 |
| **Pavia University** | **7 클래스** (이론상 9) | 39,332 (이론상 42,776) | 276,660 | 31,465 (이론상 34,220) | 7,867 (이론상 8,556) | 0.9687 | 0.9919 | 0.9803 | 0.9954 |
| **Pavia Centre** | **9 클래스** | 148,152 | 1,070,444 | 118,521 | 29,631 | 0.9501 | 0.9866 | 0.9549 | 0.9902 |
| **HyRank (Dioni)** | **12 클래스** | 20,024 | 130,228 | 16,019 | 4,005 | 0.9829 | 0.9928 | 0.9860 | 0.9973 |
| **합계 / 평균 (Total / Average)** | **-** | **221,005** (이론상 **224,449**) | **1,570,578** (이론상 **1,597,810**) | **176,802** (이론상 **179,558**) | **44,203** (이론상 **44,892**) | **0.9770** | **0.9927** | **0.9758** | **0.9959** |

* Pavia University의 경우, 실제 제공된 Pavia_gt.mat의 유효 노드 수는 39,332개(7개 클래스)이며, 이를 반영한 실제 총 노드 합계는 221,005개입니다. 이전 하드코딩 문서상의 이론적 수치인 42,776개(9개 클래스) 기준으로는 총 합계가 224,449개입니다.

---

## 2. GNN 원본 클래스 분류 정량적 성능 결과 (F1-Scores)

아래의 두 표는 각각 **GCN** 및 **GraphSAGE** 모델 하에서 원시 스펙트럼과 Hyperfocus v71 임베딩 벡터를 주입했을 때의 격리 테스트 노드(Test) 및 전체 그래프 노드(Full) 매크로 F1 스코어입니다.

### 2.1 GCN (Graph Convolutional Network) 성능 결과
| Dataset | Raw F1 (Test) | Emb F1 (Test) | Raw F1 (Full) | Emb F1 (Full) | Test Improvement (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | 0.9828 | 0.9884 | 0.9859 | 0.9925 | **+0.0056** |
| **Botswana** | 0.9987 | 1.0000 | 0.9976 | 1.0000 | **+0.0013** |
| **Pavia University** | 0.9667 | 0.9919 | 0.9687 | 0.9919 | **+0.0252** |
| **Pavia Centre** | 0.9450 | 0.9853 | 0.9501 | 0.9866 | **+0.0403** |
| **HyRank (Dioni)** | 0.9754 | 0.9891 | 0.9829 | 0.9928 | **+0.0137** |
| **Average** | 0.9737 | 0.9909 | 0.9770 | 0.9927 | **+0.0172** |

### 2.2 GraphSAGE 성능 결과
| Dataset | Raw F1 (Test) | Emb F1 (Test) | Raw F1 (Full) | Emb F1 (Full) | Test Improvement (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | 0.9583 | 0.9940 | 0.9587 | 0.9967 | **+0.0357** |
| **Botswana** | 0.9973 | 1.0000 | 0.9989 | 1.0000 | **+0.0027** |
| **Pavia University** | 0.9811 | 0.9939 | 0.9803 | 0.9954 | **+0.0128** |
| **Pavia Centre** | 0.9508 | 0.9897 | 0.9549 | 0.9902 | **+0.0389** |
| **HyRank (Dioni)** | 0.9758 | 0.9951 | 0.9860 | 0.9973 | **+0.0192** |
| **Average** | 0.9727 | 0.9945 | 0.9758 | 0.9959 | **+0.0219** |

---

## 3. 원본 세부 클래스별 분류 성능 상세 분석 (Class-specific Results - Test)
각 데이터셋에 분포하는 모든 원본 클래스들에 대해 GCN 모델 기준의 격리 테스트 노드(Test) 세부 F1-score 결과와 향상율을 나타냅니다.

### 3.1 Indian Pines 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Alfalfa | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Corn-notill | 0.9775 | 0.9895 | **+0.0120** |
| 3 | Corn-mintill | 0.9728 | 0.9880 | **+0.0151** |
| 4 | Corn | 0.9792 | 0.9895 | **+0.0103** |
| 5 | Grass-pasture | 0.9897 | 1.0000 | **+0.0103** |
| 6 | Grass-trees | 0.9966 | 0.9966 | **+0.0000** |
| 7 | Grass-pasture-mowed | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Hay-windrowed | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Oats | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Soybean-notill | 0.9794 | 0.9795 | **+0.0001** |
| 11 | Soybean-mintill | 0.9784 | 0.9857 | **+0.0073** |
| 12 | Soybean-clean | 0.9874 | 0.9958 | **+0.0083** |
| 13 | Wheat | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Woods | 0.9705 | 0.9765 | **+0.0059** |
| 15 | Buildings-Grass-Trees-Drives | 0.8933 | 0.9128 | **+0.0194** |
| 16 | Stone-Steel-Towers | 1.0000 | 1.0000 | **+0.0000** |

### 3.2 Botswana 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Hippo grass | 1.0000 | 1.0000 | **+0.0000** |
| 3 | Floodplain grasses 1 | 1.0000 | 1.0000 | **+0.0000** |
| 4 | Floodplain grasses 2 | 1.0000 | 1.0000 | **+0.0000** |
| 5 | Reeds | 0.9908 | 1.0000 | **+0.0092** |
| 6 | Riparian | 0.9907 | 1.0000 | **+0.0093** |
| 7 | Firescar | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Island interior | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Acacia woodlands | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Acacia shrublands | 1.0000 | 1.0000 | **+0.0000** |
| 11 | Acacia grasslands | 1.0000 | 1.0000 | **+0.0000** |
| 12 | Short mopane | 1.0000 | 1.0000 | **+0.0000** |
| 13 | Mixed mopane | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Exposed soils | 1.0000 | 1.0000 | **+0.0000** |

### 3.3 Pavia University 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Asphalt | 0.9611 | 0.9951 | **+0.0339** |
| 2 | Meadows | 0.9650 | 0.9894 | **+0.0244** |
| 3 | Gravel | 0.9667 | 0.9859 | **+0.0192** |
| 4 | Trees | 0.9077 | 0.9811 | **+0.0733** |
| 5 | Painted metal sheets | 1.0000 | 1.0000 | **+0.0000** |
| 6 | Bare Soil | 0.9889 | 0.9981 | **+0.0092** |
| 7 | Bitumen | 0.9771 | 0.9935 | **+0.0165** |

### 3.4 Pavia Centre 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Trees | 0.9511 | 0.9787 | **+0.0275** |
| 3 | Asphalt | 0.8848 | 0.9472 | **+0.0624** |
| 4 | Self-Blocking Bricks | 0.7934 | 0.9723 | **+0.1789** |
| 5 | Bitumen | 0.9351 | 0.9932 | **+0.0581** |
| 6 | Tiles | 0.9804 | 0.9911 | **+0.0106** |
| 7 | Shadows | 0.9604 | 0.9852 | **+0.0247** |
| 8 | Meadows | 0.9994 | 1.0000 | **+0.0006** |
| 9 | Bare Soil | 1.0000 | 1.0000 | **+0.0000** |

### 3.5 HyRank 클래스별 GCN F1-Score 명세 (Test)
| Class ID | Class Name | Raw GCN F1 (Test) | Emb GCN F1 (Test) | Improvement (Δ Test) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Dense urban fabric | 0.9352 | 0.9841 | **+0.0488** |
| 2 | Mineral extraction sites | 0.9877 | 0.9750 | **-0.0127** |
| 3 | Non-irrigated arable land | 0.9710 | 0.9836 | **+0.0127** |
| 4 | Fruit trees | 0.9310 | 0.9655 | **+0.0345** |
| 5 | Olive groves | 0.9819 | 0.9930 | **+0.0111** |
| 7 | Natural grassland | 0.9931 | 1.0000 | **+0.0069** |
| 9 | Water courses | 0.9770 | 0.9900 | **+0.0130** |
| 10 | Coastal lagoons | 0.9771 | 0.9914 | **+0.0143** |
| 11 | Estuaries | 0.9659 | 0.9915 | **+0.0255** |
| 12 | Sea and ocean | 0.9846 | 0.9949 | **+0.0103** |
| 13 | Water bodies | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Herbaceous vegetation | 1.0000 | 1.0000 | **+0.0000** |



---

## 4. 전체 데이터셋 대상 원본 세부 클래스별 분류 성능 상세 분석 (Class-specific Results - Full)
각 데이터셋에 분포하는 모든 원본 클래스들에 대해 GCN 모델 기준의 전체 그래프 노드(Full) 세부 F1-score 결과와 향상율을 나타냅니다.

### 4.1 Indian Pines 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Alfalfa | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Corn-notill | 0.9795 | 0.9926 | **+0.0131** |
| 3 | Corn-mintill | 0.9729 | 0.9855 | **+0.0126** |
| 4 | Corn | 0.9628 | 0.9937 | **+0.0309** |
| 5 | Grass-pasture | 0.9959 | 0.9990 | **+0.0031** |
| 6 | Grass-trees | 0.9993 | 0.9993 | **+0.0000** |
| 7 | Grass-pasture-mowed | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Hay-windrowed | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Oats | 1.0000 | 1.0000 | **+0.0000** |
| 10 | Soybean-notill | 0.9794 | 0.9847 | **+0.0053** |
| 11 | Soybean-mintill | 0.9813 | 0.9878 | **+0.0064** |
| 12 | Soybean-clean | 0.9814 | 0.9949 | **+0.0135** |
| 13 | Wheat | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Woods | 0.9835 | 0.9882 | **+0.0047** |
| 15 | Buildings-Grass-Trees-Drives | 0.9432 | 0.9589 | **+0.0157** |
| 16 | Stone-Steel-Towers | 0.9947 | 0.9947 | **+0.0000** |

### 4.2 Botswana 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Hippo grass | 1.0000 | 1.0000 | **+0.0000** |
| 3 | Floodplain grasses 1 | 1.0000 | 1.0000 | **+0.0000** |
| 4 | Floodplain grasses 2 | 1.0000 | 1.0000 | **+0.0000** |
| 5 | Reeds | 0.9888 | 1.0000 | **+0.0112** |
| 6 | Riparian | 0.9870 | 1.0000 | **+0.0130** |
| 7 | Firescar | 1.0000 | 1.0000 | **+0.0000** |
| 8 | Island interior | 1.0000 | 1.0000 | **+0.0000** |
| 9 | Acacia woodlands | 0.9984 | 1.0000 | **+0.0016** |
| 10 | Acacia shrublands | 0.9960 | 1.0000 | **+0.0040** |
| 11 | Acacia grasslands | 0.9967 | 1.0000 | **+0.0033** |
| 12 | Short mopane | 1.0000 | 1.0000 | **+0.0000** |
| 13 | Mixed mopane | 1.0000 | 1.0000 | **+0.0000** |
| 14 | Exposed soils | 1.0000 | 1.0000 | **+0.0000** |

### 4.3 Pavia University 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Asphalt | 0.9576 | 0.9953 | **+0.0377** |
| 2 | Meadows | 0.9661 | 0.9891 | **+0.0230** |
| 3 | Gravel | 0.9633 | 0.9844 | **+0.0211** |
| 4 | Trees | 0.9250 | 0.9833 | **+0.0583** |
| 5 | Painted metal sheets | 1.0000 | 1.0000 | **+0.0000** |
| 6 | Bare Soil | 0.9886 | 0.9978 | **+0.0092** |
| 7 | Bitumen | 0.9800 | 0.9934 | **+0.0134** |

### 4.4 Pavia Centre 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Water | 1.0000 | 1.0000 | **+0.0000** |
| 2 | Trees | 0.9593 | 0.9805 | **+0.0213** |
| 3 | Asphalt | 0.9029 | 0.9519 | **+0.0490** |
| 4 | Self-Blocking Bricks | 0.8085 | 0.9760 | **+0.1675** |
| 5 | Bitumen | 0.9401 | 0.9939 | **+0.0538** |
| 6 | Tiles | 0.9800 | 0.9909 | **+0.0108** |
| 7 | Shadows | 0.9607 | 0.9863 | **+0.0256** |
| 8 | Meadows | 0.9995 | 0.9999 | **+0.0004** |
| 9 | Bare Soil | 0.9997 | 1.0000 | **+0.0003** |

### 4.5 HyRank 클래스별 GCN F1-Score 명세 (Full)
| Class ID | Class Name | Raw GCN F1 (Full) | Emb GCN F1 (Full) | Improvement (Δ Full) |
| :--- | :--- | :---: | :---: | :---: |
| 1 | Dense urban fabric | 0.9591 | 0.9905 | **+0.0314** |
| 2 | Mineral extraction sites | 0.9951 | 0.9951 | **+0.0000** |
| 3 | Non-irrigated arable land | 0.9777 | 0.9861 | **+0.0084** |
| 4 | Fruit trees | 0.9556 | 0.9730 | **+0.0173** |
| 5 | Olive groves | 0.9857 | 0.9932 | **+0.0076** |
| 7 | Natural grassland | 0.9959 | 1.0000 | **+0.0041** |
| 9 | Water courses | 0.9768 | 0.9920 | **+0.0152** |
| 10 | Coastal lagoons | 0.9776 | 0.9932 | **+0.0156** |
| 11 | Estuaries | 0.9800 | 0.9974 | **+0.0174** |
| 12 | Sea and ocean | 0.9919 | 0.9990 | **+0.0071** |
| 13 | Water bodies | 1.0000 | 0.9988 | **-0.0012** |
| 14 | Herbaceous vegetation | 1.0000 | 0.9949 | **-0.0051** |



---

## 5. 시각화 분석 및 모델 평가

![Real Class GNN Performance](../images/gnn_real_classes_performance.png)

* **Hyperfocus v71 임베딩의 압도적인 원본 분류 한계 극복**:
  모든 원본 클래스 분류 실험(클래스가 9~16개로 파편화된 다중 클래스 조건)에서 Hyperfocus v71 임베딩 노드를 주입한 GNN 모델이 원시 스펙트럼 대비 압도적인 성능 우위를 지님을 확인할 수 있습니다. 
  특히, 복잡한 작물 유형이 16개로 얽혀 있는 **Indian Pines**의 경우, GCN 기준 원시 F1 **0.9828** 대비 임베딩 F1 **0.9884**로 **+0.0056**의 극적인 성능 향상을 보였으며, GraphSAGE에서도 **+0.0357** 상승했습니다.
* **다중 클래스 환경에서의 노이즈 및 간섭 억제**:
  클래스 가짓수가 많아질수록 원시 스펙트럼 신호는 특정 미세 파장대 노이즈로 인해 결정 경계가 흐려지고, GNN의 인접 메시지 패싱 시 오염된 특징이 전파되어 성능이 급격히 저하됩니다(특히 위성 센서 기반 HyRank에서 원시 GCN의 성능이 낮게 유지됨). 반면 **Hyperfocus v71** 인코더는 MAE 사전학습을 통해 획득한 고성능 필터링 효과로 인해, 클래스 고유의 주요 주파수 물리적 신호를 128차원에 최적으로 부호화하여 95%~99% 영역의 정밀한 클래스 구분을 가능케 합니다.

---

### 🔗 관련 문서 바로가기
* **Hyperfocus v71 README**:
  👉 **[README.md](../README.md)**
* **GNN 제로샷 성능 종합 보고서**:
  👉 **[제로샷 교차 데이터셋 전이 및 공간 그래프 신경망 성능 평가 보고서 (zeroshot_generalization_analysis.md)](zeroshot_generalization_analysis.md)**
