# End-to-End Cultural-Context Pipeline Validation Report

**System**: Drishtikon Multi-Stage Cultural Context Pipeline  
**Modules Validated**:
- `src/cultural/context_builder.py` (`sanitize_retrieval_query` and `CulturalContextBuilder`)
- `src/cultural/context_retriever.py` (`CulturalRetriever`)
- `src/cultural/category_detector.py` (`detect_categories`)
- `src/cultural/drishtikon_retriever.py` (`DrishtikonRetriever`)
- `data/cultural/cultural_knowledge.json` (Project knowledge base)
- `data/cultural/drishtikon/processed/drishtikon_knowledge.json` (64,288 indexed records)  
**Execution Type**: Strictly Read-Only, Deterministic End-to-End Pipeline Execution  
**VLM Inference Launched**: **Strictly 0 calls**  
**Source Code / Configuration / Knowledge File Modifications**: **Strictly 0**  
**Report Location**: `reports/cultural_context_pipeline_validation.md`  
**Execution Date**: 2026-10-19  

---

## 1. Executive Summary & Audit Safeguards

This report provides a deterministic, end-to-end empirical validation of the Drishtikon cultural context pipeline. Rather than testing components in isolation, this audit traces how raw OCR text travels through query sanitization, category detection, project-level RAG (`PROJECT_JSON`), and DRISHTIKON external retrieval, culminating in the assembled prompt context injected into the vision-language model (`Qwen2.5-VL-3B-Instruct`).

### 1.1 Read-Only Input Files Inspected
1. `src/cultural/context_builder.py` (Context aggregation and query sanitization)
2. `src/cultural/context_retriever.py` (Project-level retriever and ChromaDB fallback)
3. `src/cultural/category_detector.py` (Rule-based keyword category detector)
4. `src/cultural/drishtikon_retriever.py` (TF-IDF vectorizer and cosine ranking)
5. `data/cultural/cultural_knowledge.json` (19 categories, 3,425 base knowledge statements)
6. `data/cultural/drishtikon/processed/drishtikon_knowledge.json` (64,288 external MCQ records)
7. `results/evaluation_twostage_v4/comparison_results.csv` (Saved evaluation OCR and diagnostics)
8. `reports/drishtikon_sanitization_benchmark_expanded.md` (56-query benchmark baseline)
9. `reports/drishtikon_sanitization_annotation_audit.md` (Forensic annotation audit)

### 1.2 Pipeline Architecture Overview

```
                      Raw Meme OCR Text
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   sanitize_retrieval_query()        detect_categories()
               │                             │
               ▼                             ▼
      Sanitized Query Text          Detected Categories (e.g. ['family'])
               │                             │
       ┌───────┴───────┐                     │
       ▼               ▼                     ▼
PROJECT_JSON RAG   Query Augmentation: (Sanitized + Clean Categories)
       │                                     │
       ▼                                     ▼
CulturalRetriever                     DrishtikonRetriever
(cultural_knowledge.json)            (TF-IDF Index over 64,288 MCQs)
       │                                     │
       ▼                                     ▼
Project Context Bullets              Top DRISHTIKON Record (if >= 0.25)
       │                                     │
       └───────────────┬─────────────────────┘
                       ▼
         CulturalContextBuilder Assembly
                       │
                       ▼
       Final Prompt Context to Qwen2.5-VL
```

---

## 2. Comprehensive Test Case Validations

The pipeline was executed against 24 distinct test cases covering all 7 operational scenarios specified in the protocol.

---

### Case 1: Durga Puja with Noisy Handles and Watermarks (`C_SOC_08`)

- **Input Raw Text**: `"durga puja festival celebration west bengal idol @kolkata_durga fb/puja"`
- **Sanitized Retrieval Query**: `"durga puja festival celebration west bengal idol"`
- **Category Detector Output**: `['festivals', 'religion']`
- **DRISHTIKON Query Assembled**: `"durga puja festival celebration west bengal idol festivals religion"`
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0; raw query score without categories was `0.2447`, below the 0.25 threshold)
- **PROJECT_JSON Fallback Ran**: **YES** (100% active and reachable)
- **Final Sources Used**: `['PROJECT_JSON']`
- **Assembled Context Injected into Prompt**:
  ```text
  EXTERNAL CULTURAL CONTEXT:
  PROJECT CULTURAL KNOWLEDGE:
  - FESTIVALS: Diwali (deep cleaning, soan papdi forwarding, firecracker bans), Holi (color attacks, gujiya), Raksha Bandhan (pocket money draining), Eid, and Durga Puja.
  - RELIGION: Cultural traditions, rituals, festive fasting (vrat), temple visits, prasad, and respectful mythological references.
  ```
- **Context Integrity Audit**:
  - *Irrelevant Context Introduced*: **None**.
  - *Relevant Context Omitted*: **None**. Although DRISHTIKON missed due to the 0.25 threshold cutoff, `PROJECT_JSON` cleanly injected the `FESTIVALS` knowledge line with explicit mention of *"Durga Puja"*.

---

### Case 2: Numeric Collision (`10,000` Noise alongside Meaningful `420`)

- **Input Raw Text**: `"Le husband: Bhai yeh toh 420 aadmi hai 10,000 fraud"`
- **Sanitized Retrieval Query**: `"Bhai yeh toh 420 aadmi hai fraud"` (Numeral `10,000` stripped; `420` preserved)
- **Category Detector Output**: `['family']` (Triggered by colloquial address *"Bhai"*)
- **DRISHTIKON Query Assembled**: `"Bhai yeh toh 420 aadmi hai fraud family"`
- **DRISHTIKON Top Result**: `[Cultural Common Sense | Haryana] Q: How does the portrayal of traditional festivities in Haryanvi weddings... tapestry weaving family bonds and uncomplicated joy? -> Context: Simplicity and family gatherings` (Score: `0.2506`, Matches: 1)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Final Sources Used**: `['PROJECT_JSON', 'DRISHTIKON']`
- **Assembled Context Injected into Prompt**:
  ```text
  EXTERNAL CULTURAL CONTEXT:
  PROJECT CULTURAL KNOWLEDGE:
  - FAMILY: Indian family dynamics often involve strict parenting, comparison with peers (e.g., 'Sharma ji ka beta'), emphasis on respect for elders, taunts about marriage or career, close-knit joint family structures, and sibling banter (bhai/behen).

  INDIAN CULTURAL CONTEXT (DRISHTIKON):
  - [Cultural Common Sense | Haryana] How does the portrayal of traditional festivities in Haryanvi weddings, as referenced in the image, symbolize the core cultural values similar to those of a tapestry weaving family bonds and uncomplicated joy? -> Context: Simplicity and family gatherings
  ```
- **Context Integrity Audit**:
  - *Spurious Collision Avoidance*: In un-sanitized retrieval, `10,000` matched a Buddhist monastery at 10,000 feet in Arunachal Pradesh. Sanitization **100% eliminated the monastery collision**.
  - *Secondary Lexical Collision Discovered*: Because `context_builder.py` appends detected categories to the DRISHTIKON query, the token `family` (from *"Bhai"*) nudged a Haryanvi family gathering record across the threshold (Score: 0.2506). While benign, it illustrates that category query augmentation can occasionally bridge unrelated records across the 0.25 boundary.
  - *Relevant Context Omitted*: IPC Section 420 legal fraud context is missing from DRISHTIKON, but generic family banter is supplied by `PROJECT_JSON`.

---

### Case 3: Legal References, Cultural Numbers, and Historical Dates

#### Case 3a: Legal Code — Section 144 Curfew
- **Input Raw Text**: `"Police imposed Section 144 curfew in Delhi during protests"`
- **Sanitized Retrieval Query**: `"Police imposed Section 144 curfew in Delhi during protests"` (Preserved by whitelist)
- **Category Detector Output**: `['regional_culture']` (Triggered by *"Delhi"*)
- **DRISHTIKON Top Result**: `[Transport | Delhi] Which section of depicted Phase-IV was inaugurated in January 2025? -> Context: Janakpuri-Krishna Park section` (Score: `0.3133`, Matches: 1)
- **PROJECT_JSON Fallback Ran**: **YES** (`REGIONAL_CULTURE`: Delhi stereotypes)
- **Final Sources Used**: `['PROJECT_JSON', 'DRISHTIKON']`
- **Context Integrity Audit**:
  - *Irrelevant Context Introduced*: The word *"section"* matched Delhi Metro Phase-IV rail transport. This is an irrelevant lexical collision because DRISHTIKON contains no legal statutes.
  - *Relevant Context Omitted*: Legal public order curfew context is omitted from DRISHTIKON.

#### Case 3b: Constitutional Code — Article 370 Special Status
- **Input Raw Text**: `"Debate on Article 370 special status in Jammu and Kashmir"`
- **Sanitized Retrieval Query**: `"Debate on Article 370 special status in Jammu and Kashmir"` (Preserved by whitelist)
- **Category Detector Output**: `[]`
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0)
- **PROJECT_JSON Fallback Ran**: **NO** (No category matched)
- **Final Sources Used**: `[]`
- **Context Integrity Audit**:
  - *Irrelevant Context Introduced*: **None** (Clean rejection).
  - *Relevant Context Omitted*: Article 370 is not indexed in either DRISHTIKON or base project categories.

#### Case 3c: Legal Fraud — Dhara 420
- **Input Raw Text**: `"FIR filed under Dhara 420 for fraud and cheating in court"`
- **Sanitized Retrieval Query**: `"FIR filed under Dhara 420 for fraud and cheating in court"` (Preserved by whitelist)
- **Category Detector Output**: `[]`
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0)
- **PROJECT_JSON Fallback Ran**: **NO**
- **Final Sources Used**: `[]`
- **Context Integrity Audit**:
  - *Irrelevant Context Introduced*: **None** (Clean rejection).
  - *Relevant Context Omitted*: Legal IPC codes are absent from DRISHTIKON.

#### Case 3d: Emergency Dial — 108 Ambulance Service
- **Input Raw Text**: `"Ambulance emergency services dialed on 108 across states"`
- **Sanitized Retrieval Query**: `"Ambulance emergency services dialed on 108 across states"` (Preserved by whitelist)
- **Category Detector Output**: `[]`
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0)
- **PROJECT_JSON Fallback Ran**: **NO**
- **Final Sources Used**: `[]`
- **Context Integrity Audit**: Clean rejection; no spurious noise.

#### Case 3e: National Date — 15 August Independence Day
- **Input Raw Text**: `"Independence Day celebrations on 15 August Red Fort Delhi"`
- **Sanitized Retrieval Query**: `"Independence Day celebrations on 15 August Red Fort Delhi"` (Preserved by whitelist)
- **Category Detector Output**: `['regional_culture']`
- **DRISHTIKON Top Result**: `[Rituals and Ceremonies | Delhi] In which city the celebration shown in the image with flag-hoisting ceremony held at the Red Fort? -> Context: Delhi` (Score: `0.2872`, Matches: 2)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Final Sources Used**: `['PROJECT_JSON', 'DRISHTIKON']`
- **Context Integrity Audit**:
  - *Irrelevant Context Introduced*: **None**.
  - *Relevant Context Provided*: High-precision match: Red Fort flag-hoisting ceremony on Independence Day in Delhi.

#### Case 3f: National Date — 26th January Republic Day
- **Input Raw Text**: `"Republic Day parade on 26th January Rajpath New Delhi"`
- **Sanitized Retrieval Query**: `"Republic Day parade on 26th January Rajpath New Delhi"` (Preserved by whitelist)
- **Category Detector Output**: `['regional_culture']`
- **DRISHTIKON Top Result**: `[Rituals and Ceremonies | Delhi] What city, known for hosting the annual Republic Day parade during which different cultural and historical tableaux are showcased on Rajpath... -> Context: Delhi` (Score: `0.3989`, Matches: 3)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Final Sources Used**: `['PROJECT_JSON', 'DRISHTIKON']`
- **Context Integrity Audit**:
  - *Irrelevant Context Introduced*: **None**.
  - *Relevant Context Provided*: Outstanding match: Rajpath, Republic Day parade, and cultural tableaux.

---

### Case 4: Hinglish Relationship and Lockdown References

#### Case 4a: `train_5890.jpg` OCR (Modi Lockdown vs. Nibba-Nibbi Separation)
- **Input Raw Text**:
  ```text
  PM Modi Speech Live:
  Narendra Modi Says India
  Will Be Under Complete
  Lockdown From Midnight
  For 21 Days
  Nibbi
  *Nibba
  BABY HUM OR 21 DIN NAHI MIL SAKTE
  Indya Speak
  ```
- **Sanitized Retrieval Query**: `"PM Modi Speech Live: Narendra Modi Says India Will Be Under Complete Lockdown From Midnight For Days Nibbi *Nibba BABY HUM OR DIN NAHI MIL SAKTE Indya Speak"` (Isolated `21` stripped; `Nibbi`, `Nibba`, `MIL SAKTE` preserved)
- **Category Detector Output**: `['relationships']` (Matched `"nibbi"`, `"nibba"`, `"baby"`)
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Final Sources Used**: `['PROJECT_JSON']`
- **Assembled Context Injected into Prompt**:
  ```text
  EXTERNAL CULTURAL CONTEXT:
  PROJECT CULTURAL KNOWLEDGE:
  - RELATIONSHIPS: Conservative societal attitudes towards dating, sneaking calls from parents, 'nibba-nibbi' cringe teenage romance tropes, dramatic couple separation laments ('milna / nahi mil sakte' referring to meeting or seeing a partner, especially during lockdowns or curfews), breakups, friendzones, and eternal bachelorhood (singlehood).
  ```
- **Context Integrity Audit**:
  - *Relevance Evaluation*: **Flawless precision**. The prompt receives an exact explanation of the *"nibba-nibbi"* teenage romantic separation trope during lockdowns (*"milna / nahi mil sakte"*).
  - *DRISHTIKON Invariance*: DRISHTIKON correctly returns zero matches, preventing spurious historical monuments from contaminating modern internet slang.

#### Case 4b: `train_5723.jpg` OCR (The Family Man / In-Law Avoidance Gag)
- **Input Raw Text**:
  ```text
  #TheFamilyMan
  #OnlyExcusesOctober
  When Wife Starts To Tell About
  Her Parents Instead of about Her
  Sister #SalliJi
  Le Husband:
  main bead me phone karte boon
  ```
- **Sanitized Retrieval Query**: `"#TheFamilyMan #OnlyExcusesOctober When Wife Starts To Tell About Her Parents Instead of about Her Sister #SalliJi main bead me phone karte boon"` (`Le Husband:` stripped)
- **Category Detector Output**: `['family']` (Matched `"parents"`, `"wife"`)
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0)
- **PROJECT_JSON Fallback Ran**: **YES** (`FAMILY`: Indian family dynamics, respect for elders)
- **Final Sources Used**: `['PROJECT_JSON']`
- **Context Integrity Audit**:
  - *Relevance Evaluation*: Correctly routes to Indian family dynamics.
  - *Opportunity Identified*: Notice that `#TheFamilyMan` retains the `#` symbol. Because `#TheFamilyMan` is not separated into `The Family Man`, it does not participate in word-level lexical matching.

#### Case 4c: `train_6346.jpg` OCR (Quarantine Day / Arranged Marriage Taunt)
- **Input Raw Text**:
  ```text
  Quarantine Day No.5:
  Indian parents: Beta,
  ab Umar hogayi,
  Shaadi kab karoge?
  Kam se kam 30 saal ka toh time de
  ```
- **Sanitized Retrieval Query**: `"Quarantine Day No. : Indian parents: Beta, ab Umar hogayi, Shaadi kab karoge? Kam se kam saal ka toh time de"`
- **Category Detector Output**: `['family', 'marriage/wedding']` (Matched `"parents"`, `"shaadi"`)
- **DRISHTIKON Top Result**: None (Score: `0.0000`, Matches: 0)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Final Sources Used**: `['PROJECT_JSON']`
- **Assembled Context Injected into Prompt**:
  ```text
  EXTERNAL CULTURAL CONTEXT:
  PROJECT CULTURAL KNOWLEDGE:
  - FAMILY: Indian family dynamics often involve strict parenting, comparison with peers (e.g., 'Sharma ji ka beta'), emphasis on respect for elders, taunts about marriage or career, close-knit joint family structures, and sibling banter (bhai/behen).
  - MARRIAGE/WEDDING: Big fat Indian weddings, arranged marriage matrimonial Biodata, nosy relatives asking 'shaadi kab karoge?', buffet food rush, dowry and extravagant ceremonies.
  ```
- **Context Integrity Audit**:
  - *Relevance Evaluation*: **100% factual alignment**. Injects nosy relatives asking *"shaadi kab karoge?"* and parental expectations.

---

### Case 5: Meme Boilerplate, Social Handles, and Clean Cultural Entities

| Subcase | Input Entity / Noise Condition | Sanitized Query | Detected Categories | DRISHTIKON Top Result | Score | Sources Used | Context Quality Assessment |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **5a** | Kedarnath Temple (Clean) | `kedarnath temple uttarakhand pilgrimage` | `['religion']` | `[Cultural Common Sense \| Uttarakhand] Kedarnath Temple` | **0.3849** | `['PROJECT_JSON', 'DRISHTIKON']` | **Ideal dual injection**: Architectural MCQ from DRISHTIKON + religious customs from Base RAG. |
| **5b** | Kedarnath Temple (*"Me and the boys"*) | `at kedarnath temple uttarakhand pilgrimage` | `['religion']` | `[Cultural Common Sense \| Uttarakhand] Kedarnath Temple` | **0.3849** | `['PROJECT_JSON', 'DRISHTIKON']` | Boilerplate stripped; identical high-confidence score and context as clean query. |
| **5c** | Badami Caves (`backup: @... ig/...`) | `badami rock cut cave temples karnataka` | `[]` | `[History \| Karnataka] Chalukya rock-cut cave temples` | **0.4942** | `['DRISHTIKON']` | Handles stripped; exact architectural context delivered cleanly. |
| **5d** | Shah Rukh Khan (Clean) | `shah rukh khan king of bollywood actor` | `['Bollywood']` | `[Personalities \| Delhi] King of Bollywood Shah Rukh Khan` | **0.6570** | `['PROJECT_JSON', 'DRISHTIKON']` | Dual injection: Bollywood superstar persona from Base RAG + Delhi origin from DRISHTIKON. |
| **5e** | Chinnaswamy Stadium (Clean) | `chinnaswamy stadium bangalore cricket karnataka` | `['cricket', 'regional_culture']` | `[Sports \| Karnataka] Chinnaswamy Stadium cricket venue` | **0.3315** | `['PROJECT_JSON', 'DRISHTIKON']` | Dual injection: Cricket religion / RCB memes from Base RAG + stadium location from DRISHTIKON. |

---

### Case 6: DRISHTIKON Has No Match, but PROJECT_JSON Should Supply Context

This scenario evaluates whether the pipeline reliably injects universal cultural knowledge for social and conversational memes that do not correspond to historical monuments:

#### Case 6a: Indian Parenting & Academic Pressure
- **Input Raw Text**: `"Indian parents: Beta engineering kar lo shaadi kab karoge Sharma ji ka beta"`
- **Category Detector Output**: `['family', 'education', 'marriage/wedding', 'social_norms']`
- **DRISHTIKON Match Count**: 0 (Score: `0.0000`)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Sources Used**: `['PROJECT_JSON']`
- **Injected Context**: Injects comprehensive coverage of parental engineering pressure (`EDUCATION`), arranged marriage matrimonial anxiety (`MARRIAGE/WEDDING`), and *"Sharma ji ka beta / Log kya kahenge"* peer comparison (`FAMILY`, `SOCIAL_NORMS`).

#### Case 6b: College Life & Exam Anxiety
- **Input Raw Text**: `"Bhai yeh kya scene hai pura college bunk maar ke ghum rahe exam season"`
- **Category Detector Output**: `['family', 'college', 'JEE/exams']`
- **DRISHTIKON Match Count**: 0 (Score: `0.0000`)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Sources Used**: `['PROJECT_JSON']`
- **Injected Context**: Injects 75% mandatory attendance rules, KT/backlogs, and campus placement anxiety (`COLLEGE`) along with coaching hub competition (`JEE/EXAMS`).

#### Case 6c: Street Food & Chai Tapri Culture
- **Input Raw Text**: `"Chai tapri pe dost ke sath gossips and samosa break daily life"`
- **Category Detector Output**: `['food']`
- **DRISHTIKON Match Count**: 0 (Score: `0.0000`)
- **PROJECT_JSON Fallback Ran**: **YES**
- **Sources Used**: `['PROJECT_JSON']`
- **Injected Context**: Injects cutting chai, samosa break, and street food obsession (`FOOD`).

---

### Case 7: Cases Where NO Cultural Context Should Be Added (Pure Noise / Non-Cultural)

This scenario stress-tests the negative-control boundary. For non-cultural western memes, adult spam, and generic numbers, the pipeline must add **zero cultural context**:

| Subcase | Input Query / Meme Sample | Detected Categories | DRISHTIKON Matches | PROJECT_JSON Ran | Sources Used | Injected Prompt Context | Audit Result |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **7a** | `train_5088.jpg` OCR (Brexit / Trump debate) | `[]` | 0 (0.0000) | **NO** | `[]` | `""` (Empty) | **CORRECT (Zero noise)** |
| **7b** | `train_5266.jpg` OCR (`hot Indian milfs` spam) | `[]` | 0 (0.0000) | **NO** | `[]` | `""` (Empty) | **CORRECT (Zero noise)** |
| **7c** | `train_3395.jpg` OCR (Kamala birthday card) | `[]` | 0 (0.0000) | **NO** | `[]` | `""` (Empty) | **CORRECT (Zero noise)** |
| **7d** | `train_3280.jpg` OCR (Cyanide & Happiness comic) | `[]` | 0 (0.0000) | **NO** | `[]` | `""` (Empty) | **CORRECT (Zero noise)** |
| **7e** | Numeric noise (`10,000 cash prize 50000 views`) | `[]` | 0 (0.0000) | **NO** | `[]` | `""` (Empty) | **CORRECT (Zero noise)** |

**Attestation on Negative Controls**:
Across all 5 negative control cases, `sources_used` was strictly empty `[]`, and the prompt context returned was strictly empty `""`. Zero irrelevant cultural bullets were injected into any non-cultural meme.

---

## 3. Comprehensive Execution Matrix (24 Test Cases)

| Case ID | Scenario Name | Sanitized Query Preview | Detected Cats | DRISHTIKON Top Score | DRISHTIKON Top Match | Proj Ran? | Sources Used | Prompt Context Quality |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| `1` | Durga Puja with Handles (`C_SOC_08`) | `durga puja festival celebration...` | `['festivals', 'religion']` | 0.0000 | None (Score 0.2447 < 0.25) | **YES** | `['PROJECT_JSON']` | **Protected**: Durga Puja injected via `PROJECT_JSON`. |
| `2` | Numeric Collision (10,000 vs 420) | `Bhai yeh toh 420 aadmi hai fraud` | `['family']` | 0.2506 | Haryanvi wedding family bonds | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | **Protected**: 10,000 monastery killed; family context added. |
| `3a` | Legal Code: Section 144 Curfew | `Police imposed Section 144...` | `['regional_culture']` | 0.3133 | Delhi Metro Section Phase-IV | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | Spurious DRISHTIKON match on word *"section"*. |
| `3b` | Legal Code: Article 370 | `Debate on Article 370...` | `[]` | 0.0000 | None | NO | `[]` | Clean rejection (not in DRISHTIKON). |
| `3c` | Legal Code: Dhara 420 | `FIR filed under Dhara 420...` | `[]` | 0.0000 | None | NO | `[]` | Clean rejection (not in DRISHTIKON). |
| `3d` | Emergency: 108 Ambulance | `Ambulance emergency... on 108` | `[]` | 0.0000 | None | NO | `[]` | Clean rejection. |
| `3e` | National Date: 15 August | `Independence Day... 15 August...` | `['regional_culture']` | 0.2872 | Red Fort flag-hoisting ceremony | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | **High precision**: Flag-hoisting at Red Fort. |
| `3f` | National Date: 26th January | `Republic Day parade 26th January` | `['regional_culture']` | 0.3989 | Rajpath Republic Day parade | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | **High precision**: Rajpath parade tableaux. |
| `4a` | Lockdown Romance (`train_5890`) | `PM Modi Speech... BABY HUM OR...` | `['relationships']` | 0.0000 | None | **YES** | `['PROJECT_JSON']` | **Flawless**: Injects lockdown separation trope. |
| `4b` | In-law Gag (`train_5723`) | `#TheFamilyMan... main bead me...` | `['family']` | 0.0000 | None | **YES** | `['PROJECT_JSON']` | Injects family dynamics and elder respect. |
| `4c` | Marriage Taunt (`train_6346`) | `Quarantine Day... Shaadi kab...` | `['family', 'marriage/wedding']` | 0.0000 | None | **YES** | `['PROJECT_JSON']` | **Flawless**: Injects *"shaadi kab karoge?"* |
| `5a` | Clean Monument: Kedarnath | `kedarnath temple uttarakhand...` | `['religion']` | 0.3849 | Kedarnath temple architecture | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | Dual injection: Architecture + religious customs. |
| `5b` | Boilerplate + Kedarnath | `at kedarnath temple uttarakhand...`| `['religion']` | 0.3849 | Kedarnath temple architecture | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | Identical high score after stripping boilerplate. |
| `5c` | Handles + Badami Caves | `badami rock cut cave temples...` | `[]` | 0.4942 | Chalukya rock-cut caves | NO | `['DRISHTIKON']` | Pure DRISHTIKON architectural context. |
| `5d` | Personality: Shah Rukh Khan | `shah rukh khan king of bollywood` | `['Bollywood']` | 0.6570 | King of Bollywood SRK | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | Dual injection: Persona + Delhi roots. |
| `5e` | Landmark: Chinnaswamy Stadium | `chinnaswamy stadium bangalore...` | `['cricket', 'regional_culture']` | 0.3315 | Chinnaswamy Stadium venue | **YES** | `['PROJECT_JSON', 'DRISHTIKON']` | Dual injection: Cricket religion + stadium venue. |
| `6a` | Parenting Trope (Sharma ji) | `Indian parents: Beta engineering...`| `['family', 'education', ...]` | 0.0000 | None | **YES** | `['PROJECT_JSON']` | Injects parenting, marriage, and social norms. |
| `6b` | College Bunking Slang | `Bhai yeh kya scene hai college...`| `['family', 'college', ...]` | 0.0000 | None | **YES** | `['PROJECT_JSON']` | Injects 75% attendance rules and backlog anxiety. |
| `6c` | Chai Tapri Culture | `Chai tapri pe dost ke sath...` | `['food']` | 0.0000 | None | **YES** | `['PROJECT_JSON']` | Injects street food and cutting chai. |
| `7a` | Text Meme (`train_5088`) | `THIS is not COURAGE... brexit` | `[]` | 0.0000 | None | NO | `[]` | **Zero noise**: Context is empty string. |
| `7b` | Clickbait Spam (`train_5266`) | `hot Indian milfs` | `[]` | 0.0000 | None | NO | `[]` | **Zero noise**: Context is empty string. |
| `7c` | Birthday Card (`train_3395`) | `HAPPY BIRTHDAY FROM KAMALA!` | `[]` | 0.0000 | None | NO | `[]` | **Zero noise**: Context is empty string. |
| `7d` | Webcomic (`train_3280`) | `Do you support pedopihila Reddit` | `[]` | 0.0000 | None | NO | `[]` | **Zero noise**: Context is empty string. |
| `7e` | Isolated Numeric Noise | `Earned cash prize with views...` | `[]` | 0.0000 | None | NO | `[]` | **Zero noise**: Context is empty string. |

---

## 4. Key Architectural Insights & Identified Trade-Offs

1. **Category Augmentation Mechanics (`query_parts`)**:
   - In `src/cultural/context_builder.py` (lines 150–154), the DRISHTIKON query is formed by concatenating the sanitized text with any detected categories (`clean_cats`).
   - *Positive Effect*: For specific monuments like Chinnaswamy Stadium (`+ ['cricket', 'regional_culture']`), category tokens reinforce the topic, boosting relevance and state alignment.
   - *Side-Effect*: In Case 2 (`"Bhai yeh toh 420 aadmi hai 10,000 fraud"`), colloquial address *"Bhai"* triggered category `'family'`, which appended `"family"` to the DRISHTIKON query. This produced a border match (`0.2506`) to a Haryanvi family wedding question. While harmless, developers should be aware that category augmentation can introduce subtle lexical pull.

2. **ChromaDB vs. Fallback Keyword Matcher in `CulturalRetriever`**:
   - In `src/cultural/context_retriever.py`, if ChromaDB sentence-transformer embeddings are not initialized, `self.use_chromadb` evaluates to `False`, activating the keyword matcher `CulturalKnowledgeBase`.
   - The keyword matcher reliably detected categories for all Hinglish relationship, family, wedding, college, and festival memes.
   - For legal and political phrases with no mapped category (e.g. *"Article 370"*, *"108 ambulance"*), `CulturalRetriever` returns `"- No specific cultural reference detected."`, cleanly preventing hallucinated prompt bullets.

3. **Protection of Modern Pop-Culture & Conversational Memes**:
   - Memes like `train_5890.jpg` (*"nibba-nibbi"* lockdown separation) and `train_6346.jpg` (*"shaadi kab karoge?"*) rely entirely on `PROJECT_JSON`.
   - DRISHTIKON correctly returned 0 matches for these memes, ensuring that ancient monuments or temples are never forced into modern internet satire.

---

## 5. Attestation & Execution Confirmation

- **Report Filename**: `reports/cultural_context_pipeline_validation.md`
- **CPU / VLM Inference Calls Launched**: **0**
- **Existing Reports or Evaluation Directories Modified**: **0**
- **Pipeline Code / Knowledge Files Modified**: **0**
- **Status**: Validation complete. Standing by for instructions.
