# CAT Intelligence Platform — shared build contract (read fully before working)

Root: `cat-platform/` (inside repo `Weekend-Projects`). Static web app that MUST work by
double-clicking `index.html` (file://) and on any static host. Therefore: no ES modules, no
fetch() of local files, no build step, no CDN. Plain `<script src>` tags; data files are JS
that assign into the global `window.CATDATA`. Vendored libs only (in `vendor/`).

## Hard integrity rules (apply to every agent)
- NEVER fabricate a historical CAT question, answer, statistic or trend. Real PYQ text could
  not be downloaded in this environment (network blocks all exam sites), so the app ships
  with NO fabricated PYQs. Historical facts come only from cited web-search results, each
  with a source id and a reliability label. Unknown = `null` + recorded gap.
- Every generated question is ORIGINAL practice material, labelled `kind:"mock"` /
  `origin:"original"`. Do not copy or closely paraphrase real CAT / coaching questions.
- Every answer key must be independently verified (compute QA/DILR answers with a Python
  script wherever computable; re-solve logically otherwise). Record `verification`.
- No LaTeX. Use HTML: `<sup>`, `<sub>`, `&radic;`, `&times;`, `&le;`, `<table>` for data.

## Folder layout
```
cat-platform/
  index.html            css/app.css        vendor/        js/*.js
  data/00_init.js       -> window.CATDATA = {papers:[], patterns:[], ...}
  data/exam_history.js  data/sources.js  data/topic_distribution.js  data/findings.js
  data/patterns.js      data/formulas.js
  data/papers/mock1.js .. mock5.js         (each: CATDATA.papers.push({...}))
  tests/                (Playwright + data validation scripts)
  docs: PROJECT_STATE.md RESEARCH_REGISTER.md DATA_DICTIONARY.md ARCHITECTURE.md
        ANALYSIS_FINDINGS.md IMPLEMENTATION_LOG.md NEXT_SESSION.md TEST_REPORT.md README.md
```
`index.html` loads `data/00_init.js` first, then every data file, then js. A missing data file
must not break the app (guard with `|| []`).

## Taxonomy (use these exact strings for `section`, `topic`; subtopic is free text but reuse)
- VARC: `Reading Comprehension` (subtopics: Main Idea, Inference, Specific Detail, Tone & Attitude,
  Strengthen/Weaken, Structure & Function, Vocabulary in Context, Application), `Para Jumbles`,
  `Para Summary`, `Odd Sentence Out`, `Sentence Placement`
- DILR: `Arrangements`, `Games & Tournaments`, `Scheduling`, `Selection & Distribution`,
  `Tables`, `Graphs & Charts`, `Caselets`, `Routes & Networks`, `Venn Diagrams`,
  `Quantitative Reasoning`, `Hybrid Sets`
- QA: `Arithmetic` (Percentages, Profit & Loss, SI/CI, Ratio & Proportion, Mixtures & Alligation,
  Averages, Time & Work, Time Speed Distance), `Algebra` (Linear Equations, Quadratic Equations,
  Inequalities, Functions, Logarithms, Surds & Indices, Progressions, Modulus, Polynomials),
  `Geometry & Mensuration` (Triangles, Circles, Quadrilaterals & Polygons, Mensuration,
  Coordinate Geometry), `Number System` (Divisibility & Remainders, Factors, HCF & LCM,
  Digits & Bases), `Modern Math` (Permutations & Combinations, Probability, Set Theory)

## Paper schema (`CATDATA.papers[]`)
```js
{ id:"mock1", kind:"mock"|"pyq", origin:"original"|"official"|"third-party"|"user-import",
  title, year:null|2024, slot:null|1, philosophy:"…", baseline:"CAT 2024 structure",
  durationMin:120, sectionalTimeLimit:true, sources:[sourceId],
  sections:[ { id:"VARC"|"DILR"|"QA", name, timeMin:40,
    groups:[ { id:"mock1-VARC-g1", type:"passage"|"set"|"standalone",
      title, text:"<p>html passage / set description incl. tables</p>",
      meta:{ /* passage: topicArea, words, readingComplexity, mainIdea, tone
                set: dataFormat, logicalStructure, constraints, concepts, difficulty */ },
      questions:[ { id:"mock1-VARC-q1", n:1 /* 1-based within section */,
        type:"mcq"|"tita", stem:"html", options:["A text","B","C","D"] /* mcq only */,
        answer: 2 /* mcq: 0-based option index */ | "42" /* tita: string; numeric compare */,
        marks:3, negative:1 /* 0 for tita */, topic, subtopic, pattern:null|patternId,
        difficulty:"easy"|"medium"|"hard", estTimeSec:120,
        solution:"html step-by-step", altMethod:"html or ''", commonMistakes:["…"],
        concepts:["…"], distractors:"html or ''",
        verification:"computed"|"re-solved"|"unverified" } ] } ] } ] }
```
Standalone questions (QA, VA) each sit in their own group with `type:"standalone"`, `text:""`.

## Mock structure (baseline: CAT 2024 — 68 questions, 120 min, 3 sections x 40 min, +3/-1, no negative for TITA)
- VARC 24: 4 RC passages (450-550 words, 4 Qs each, all MCQ) + 8 VA: 2 Para Summary (MCQ),
  2 Para Jumbles (TITA, answer like "3142"), 2 Odd Sentence Out (TITA, answer "4"),
  2 Sentence Placement (MCQ).
- DILR 22: 5 sets with 4,4,4,5,5 questions; ~8 TITA total.
- QA 22: Arithmetic 8, Algebra 7, Geometry & Mensuration 3, Number System 2, Modern Math 2; ~8 TITA.
- Difficulty mix by mock: M1 30/50/20 (easy/med/hard), M2 30/50/20, M3 20/50/30, M4 10/45/45, M5 25/50/25.
- Uniqueness allocation (do not stray, avoids duplicates across mocks):

| Mock | RC passage areas | DILR set types | QA flavour |
|---|---|---|---|
| mock1 Historical Baseline | economics of cities; philosophy of science; marine biology; history of printing | Arrangements (linear), Tables, Games & Tournaments, Venn Diagrams, Caselets | standard textbook-faithful |
| mock2 Recurring Patterns | behavioural psychology; linguistics; climate policy; art history | Scheduling, Selection & Distribution, Graphs & Charts, Routes & Networks, Arrangements (circular) | classic recurring models (alligation, work-pipes, AP/GP, remainders, etc.) |
| mock3 Conceptual Depth | anthropology; mathematics & beauty; evolutionary theory; legal philosophy | Quantitative Reasoning, Hybrid Sets, Tables (missing data), Games (scoring), Arrangements (matrix) | unfamiliar twists on standard concepts |
| mock4 Advanced Difficulty | epistemology; economics of innovation; neuroscience of memory; postcolonial literature | Hybrid Sets, Routes & Networks (flows), Scheduling (constraints), Selection (optimisation), Quantitative Reasoning | multi-concept, heavy reasoning |
| mock5 Comprehensive Simulation | urban sociology; history of medicine; technology ethics; ecology | Tables, Arrangements, Venn Diagrams (4-set / maxima-minima), Games & Tournaments, Caselets | balanced, exam-realistic |

## Pattern schema (`CATDATA.patterns[]`)
```js
{ id:"QA-ARITH-ALLIGATION", name, section, topic, subtopic, description, whyItMatters,
  approach:["step…"], traps:["…"], timeSavers:["…"],
  recurrence:{ claim:"historical recurrence statement or null", evidence:[{sourceId, note}],
               confidence:"documented"|"commonly-reported"|"unverified" },
  practice:[ question objects (same schema as paper questions; ids `pat-<patternId>-1..`;
             for DILR/RC patterns use `group` objects instead: practiceGroups:[group]) ] }
```

## Other data
- `CATDATA.sources[] = {id, name, url, year, accessed:"2026-11-15", type, reliability:
  "official"|"reputable-third-party"|"reconstructed"|"unverified", verification, limitations}`
- `CATDATA.examHistory[] = {year, mode:"paper"|"computer", conductingIIM, slots, durationMin,
  totalQuestions, sections:[{name, questions, mcq, tita, timeMin}], sectionalTimeLimit,
  marking:{correct, wrongMcq, wrongTita}, optionsPerMcq, registered, appeared, changes:"…",
  notes, sources:[id], verification}` (null where not sourced)
- `CATDATA.topicDistribution[] = {year, slot|null, section, topic, questions, difficulty|null, sources, verification}`
- `CATDATA.findings[] = {id, title, statement, evidence:[{sourceId,note}], kind:"historical-fact"|"reported-trend"|"hypothesis", confidence}`
- `CATDATA.formulas[] = {id, section:"QA", topic, subtopic, title, formula:"html", note, example}`
