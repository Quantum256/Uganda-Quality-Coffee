# Predicting Specialty Arabica Coffee Quality Using Ensemble Machine Learning

## Dataset

**CQI Arabica Quality Database (May 2023)**  
Source: [Kaggle – Coffee Quality Data CQI](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi)  
Sub-sample used: `df\\\_arabica\\\_clean.csv` — 207 records × 41 variables

**Target Variable:** `Total Cup Points` (continuous regression) and `Specialty Grade` (binary: 1 if Total Cup Points ≥ 80, else 0)

\---

## Theoretical Frameworks

This study is grounded in three complementary theoretical frameworks, applied analogously to how the UTAUT model was used by Kato et al. (2026) to organise and justify feature selection in MFA adoption prediction.

### 5.1 Precision Agriculture Theory

Precision agriculture argues that within-lot variability in agronomic and post-harvest inputs is measurable, modelable, and actionable. Applied here, Total Cup Score is not random but is a predictable function of measurable inputs — altitude, processing method, moisture, and sensory attributes — that can be quantified and acted upon to guide quality upgrading.

### 5.2 Data-Driven Quality Intelligence (DDQI) Framework

The DDQI framework posits that expert evaluator knowledge (Q Grader cupping scores encoded in the CQI database) can be extracted and deployed through machine learning, enabling quality prediction without physical expert presence. The ten CQI sensory dimensions represent structured domain knowledge that ensemble ML can encode and regularise.

### 5.3 Global Value Chain (GVC) Theory

GVC theory (Gereffi et al., 2005) provides the economic rationale: specialty-grade certification is the mechanism by which producing-country actors capture value. A predictive model that identifies high-scoring lots and explains the drivers of quality translates into actionable intelligence for process upgrading (post-harvest) and product upgrading (consistent specialty access).

\---

## Construct Mapping Table

Modelled after the UTAUT variable-to-construct mapping in Kato et al. (2026), the table below assigns each raw CQI dataset column to one of **six domain-specific quality constructs** derived from Precision Agriculture Theory, the DDQI Framework, and GVC Theory. Each mapping is justified by its theoretical role in determining `Total Cup Points`.

> \\\*\\\*Reading the table:\\\*\\\* \\\*Construct\\\* = the theoretical grouping (analogous to UTAUT's Performance Expectancy, Social Influence, etc.). \\\*Raw Column(s)\\\* = the actual dataset field name(s). \\\*Theoretical Justification\\\* = why this variable is expected, by theory, to influence the target variable.

\---

### Construct 1 — Terroir \& Agronomic Origin

|Raw Column|Type|Role in Construct|Theoretical Justification|
|-|-|-|-|
|`Country of Origin`|Categorical|Primary terroir identifier|GVC Theory: origin determines the market segment and buyer expectations a coffee can access. Precision Agriculture Theory: country encodes broad agro-climatic conditions (rainfall patterns, soil mineralogy, disease pressure) that systematically shape cup quality.|
|`Region`|Categorical|Sub-national terroir signal|Precision Agriculture: within-country variation in altitude, humidity, and shade coverage produce measurable cup-score differences; region captures this sub-national agro-climatic heterogeneity.|
|`Altitude`|Continuous (text-parsed to median metres)|Microclimate proxy|Precision Agriculture Theory: altitude is the single most documented agronomic predictor of Arabica quality — higher elevation slows bean maturation, increases density, and concentrates sugars and acids. Expected positive relationship with Total Cup Points.|
|`Harvest Year`|Categorical/Ordinal|Vintage and climate signal|DDQI Framework: year-on-year climatic variation (El Niño, rainfall anomalies) systematically shifts cup profiles; including harvest year controls for temporal quality variation not captured by stable terroir variables.|
|`Variety`|Categorical (48 levels)|Genetic quality potential|Precision Agriculture: variety determines the biochemical ceiling for cup quality (e.g., Geisha vs. Bourbon vs. Catimor). GVC Theory: certain varieties command premium buyer recognition independent of processing.|

\---

### Construct 2 — Post-Harvest Processing Quality

|Raw Column|Type|Role in Construct|Theoretical Justification|
|-|-|-|-|
|`Processing Method`|Categorical (Washed, Natural, Honey, Semi-washed)|Primary post-harvest pathway|Precision Agriculture Theory: processing method is the most controllable post-harvest decision and directly determines fermentation chemistry, mucilage exposure, and drying dynamics — all of which alter sensory attributes, particularly Flavor and Aftertaste. GVC Theory: processing method is a key product-upgrading lever for producers aiming at specialty segments.|
|`Moisture Percentage`|Continuous|Post-harvest quality control indicator|DDQI Framework: moisture content is the most commonly monitored objective quality indicator in coffee grading. Deviation from the optimal 10–12% range (SCA standard) signals deficient drying (too wet → ferment/mould risk; too dry → brittleness, off-flavours). It is a measurable, actionable precision agriculture output variable.|
|`Color`|Categorical|Bean appearance indicator|Precision Agriculture: bean colour (green, bluish-green, brownish) reflects drying uniformity and storage conditions; abnormal colour correlates with reduced cup quality and is a standard physical quality marker in CQI evaluation.|
|`Clean Cup`|Continuous (0–10)|Processing cleanliness score|DDQI Framework: Clean Cup directly measures the absence of processing-induced taints and defects in the liquor; it is the sensory operationalisation of processing quality, the Q Grader's evaluation of fermentation control and drying hygiene.|
|`Uniformity`|Continuous (0–10)|Lot consistency indicator|Precision Agriculture: Uniformity scores reflect within-lot consistency — whether all five cups evaluated score equivalently. Low uniformity signals sorting failures, mixed-variety contamination, or uneven drying, all actionable at the farm or mill level.|

\---

### Construct 3 — Sensory Complexity \& Flavour Profile

|Raw Column|Type|Role in Construct|Theoretical Justification|
|-|-|-|-|
|`Aroma`|Continuous (0–10)|Volatile compound expression|DDQI Framework: Aroma is evaluated pre- and post-break and captures the richness and complexity of volatile organic compounds. It is one of the most heavily weighted dimensions in Q Grader evaluation and correlates strongly with Total Cup Points.|
|`Flavor`|Continuous (0–10)|Core taste expression|DDQI/Precision Agriculture: Flavor is the composite in-cup taste perception, integrating acidity, sweetness, body, and aftertaste into a single holistic score. It is the most direct sensory operationalisation of the quality construct and is expected to be the strongest individual predictor of Total Cup Points.|
|`Aftertaste`|Continuous (0–10)|Post-swallow quality retention|DDQI: Aftertaste measures the duration and quality of flavour lingering after swallowing. Processing method (especially natural/honey) is known to extend positive aftertaste through residual sugar and volatile retention.|
|`Body`|Continuous (0–10)|Mouthfeel and viscosity|Precision Agriculture: Body reflects the weight and texture of the coffee liquor, influenced by bean density (itself altitude-determined) and extraction efficiency. It interacts with processing method and altitude in determining the overall sensory experience.|
|`Balance`|Continuous (0–10)|Sensory harmony|DDQI Framework: Balance captures the harmony among all sensory dimensions evaluated. A coffee with high individual attribute scores but poor balance is penalised by Q Graders. It operationalises the overall coherence of the cup profile.|

\---

### Construct 4 — Defect Burden

|Raw Column|Type|Role in Construct|Theoretical Justification|
|-|-|-|-|
|`Category One Defects`|Count (integer)|Primary defect severity|Precision Agriculture Theory: Category 1 defects (full black, full sour, pods, large stones) are the most severe quality failures and lead directly to cup score reductions. They represent measurable agronomic and processing failures (disease, fermentation errors, foreign matter) that are addressable through precision farm management.|
|`Category Two Defects`|Count (integer)|Secondary defect prevalence|Precision Agriculture/DDQI: Category 2 defects (partials, floaters, shells, brokens) reflect sorting and post-harvest handling quality. High Category 2 counts signal systemic processing inefficiencies that depress cup scores through both direct taint and indirect physical-to-sensory pathways.|
|`Quakers`|Count (integer)|Under-ripe bean detection|GVC Theory: Quakers (unripened beans that fail to develop during roasting) are a selective harvesting failure. They signal inadequate selective picking practices and, by GVC theory, represent a farm-level process-upgrading gap that producers can address to access specialty markets.|
|`Defects`|Continuous (composite)|Aggregated defect signal|DDQI Framework: The composite `Defects` variable encodes the overall physical defect load of the sample. In the DDQI model, it is the objective quality-control analogue to expert sensory assessment — measurable without a Q Grader and deployable as a pre-cupping screening variable.|

\---

### Construct 5 — Brightness \& Sweetness Expression

|Raw Column|Type|Role in Construct|Theoretical Justification|
|-|-|-|-|
|`Acidity`|Continuous (0–10)|Brightness and organic acid expression|Precision Agriculture Theory: Acidity is the defining quality attribute differentiating specialty from commodity Arabica. It is altitude-determined (higher elevation → more malic and citric acid retention) and processing-sensitive (natural processing dampens perceived acidity relative to washed). Positive relationship with Total Cup Points in specialty context.|
|`Sweetness`|Continuous (0–10)|Sugar retention and fermentation control|DDQI/GVC: Sweetness scores reflect the successful retention of fruit sugars through ripeness at harvest and careful fermentation management. In GVC terms, sweetness is a specialty-market differentiator; in Precision Agriculture terms, it is an output variable of harvest timing and processing quality decisions.|
|`Overall`|Continuous (0–10)|Q Grader holistic assessment|DDQI Framework: The Overall score is the Q Grader's discretionary holistic quality assessment beyond the sum of attribute scores. It operationalises expert tacit knowledge — the intangible quality signal that makes a coffee exceptional beyond its technical attributes. As an input feature (where available pre-aggregation), it tests how well the model can learn from structured expert knowledge.|

\---

### Construct 6 — Value Chain Traceability \& Certification

|Raw Column|Type|Role in Construct|Theoretical Justification|
|-|-|-|-|
|`Certification Body`|Categorical|Quality verification institution|GVC Theory: The certifying body (e.g., SCA, CQI-licensed Q Grader) signals the governance structure of the evaluation and affects the comparability of scores across records. In value chain terms, certification body is a chain governance indicator that determines buyer trust in the score.|
|`Producer`|Categorical|Supply chain actor identification|GVC Theory: Producer identity encodes farm-level practices, market relationships, and historical quality track records that are not captured by agronomic variables alone. It is a relational asset in the coffee GVC — known producers command buyer recognition.|
|`Country of Origin`|(also in Construct 1)|Market access category|GVC Theory: Country of origin is simultaneously a terroir signal (Construct 1) and a value chain positioning variable — origins with established specialty reputations (e.g., Ethiopia, Kenya) access different buyer segments than origins building reputation (e.g., Uganda, Thailand). This dual role mirrors UTAUT's use of demographic factors as both descriptors and behavioral predictors.|
|`In-Country Partner`|Categorical|Export channel intermediary|GVC Theory: The in-country partner (exporter, cooperative, washing station aggregator) is a chain intermediary whose sorting, logistics, and grading practices mediate between farm-level quality and the Q Grader evaluation. Different partners impose systematically different quality floors.|
|`Number of Bags`|Integer|Lot scale|Precision Agriculture: Lot size is a quality-consistency indicator — very large lots are harder to maintain as uniform quality across the whole consignment. Smaller, traceable micro-lots are associated with specialty-grade targeting in the GVC literature.|

\---

## Construct Summary (Analogous to UTAUT Table)

|Construct|# Raw Variables Mapped|Theoretical Source|Expected Direction of Effect on Total Cup Points|
|-|-|-|-|
|**C1: Terroir \& Agronomic Origin**|5|Precision Agriculture Theory + GVC Theory|Positive (higher altitude, established origins)|
|**C2: Post-Harvest Processing Quality**|5|Precision Agriculture + DDQI|Positive (washed/honey for clean profiles; optimal moisture)|
|**C3: Sensory Complexity \& Flavour Profile**|5|DDQI Framework|Strongly positive (direct component scores)|
|**C4: Defect Burden**|4|Precision Agriculture Theory + DDQI|Negative (higher defect counts → lower TCS)|
|**C5: Brightness \& Sweetness Expression**|3|Precision Agriculture + GVC|Positive in specialty context|
|**C6: Value Chain Traceability \& Certification**|5|GVC Theory|Mixed (governance and channel moderators)|

\---

## Mapping to Engineered Features (Phase 2 Outputs)

The construct mapping directly informs the six composite engineered features defined in the preprocessing pipeline:

|Engineered Feature|Source Construct(s)|Raw Variables Used|
|-|-|-|
|`Defect Burden`|C4 — Defect Burden|`Category One Defects`, `Category Two Defects`|
|`Sensory Consistency Index`|C3 — Sensory Complexity|SD of all 10 individual sensory scores|
|`Processing Quality Proxy`|C2 — Post-Harvest Processing|`Clean Cup`, `Uniformity`, `Total Cup Points`|
|`Altitude Tier`|C1 — Terroir \& Agronomic Origin|`Altitude` (parsed to metres, then binned)|
|`Flavour-Aftertaste Composite`|C3 + C2 — Sensory × Processing|`Flavor` + `Aftertaste`|
|`Attribute Completeness`|C3 — Sensory Complexity (data quality)|Count of non-null sensory attributes|

\---

## Preprocessing Pipeline Summary

1. **Missing value handling** — Variables with >40% missingness excluded; continuous imputation via MissForest; categorical imputation via country-level mode
2. **Feature encoding** — Processing Method and Variety one-hot encoded; Altitude text-parsed to continuous median metres
3. **Feature engineering** — Six composite features derived (see table above)
4. **Target definition** — `Total Cup Points` (regression) + `Specialty Grade` binary (TCS ≥ 80) + Multiclass (Excellent/Specialty/Below Specialty)
5. **Feature scaling** — Standardisation (mean=0, SD=1) for MLP/SVM comparisons; raw features used for SHAP interpretability

\---

## References

* Gereffi, G., Humphrey, J., \& Sturgeon, T. (2005). The governance of global value chains. *Review of International Political Economy*, 12(1), 78–104.
* Kato, R., Obbo, A., \& Kimera, R. (2026). Predicting multi-factor authentication uptake using machine learning and the UTAUT framework. *Academia AI and Applications*, 2. https://doi.org/10.20935/AcadAI8107
* SCA (Specialty Coffee Association). (2015). *Cupping Protocols*. SCA Publications.
* Venkatesh, V., Morris, M. G., Davis, G. B., \& Davis, F. D. (2003). User acceptance of information technology: Toward a unified view. *MIS Quarterly*, 27(3), 425–478.


