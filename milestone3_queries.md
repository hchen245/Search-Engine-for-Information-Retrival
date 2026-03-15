# Milestone 3 Query Set

This document defines the 24 evaluation queries used for Milestone 3.

- Total queries: **24**
- Initially poor queries: **12**
- Initially good queries: **12**

## A) Initially Poor Queries (12)

| # | Query | Why this was selected as "poor" initially |
|---|---|---|
| 1 | information retrieval project | Multi-term topical query; easy to miss one term in strict AND |
| 2 | graduate student forms | Administrative intent with lexical variation (forms/documents) |
| 3 | machine learning systems | Topic query where relevant docs may contain only partial term overlap |
| 4 | data science seminar | Event-style query; sparse co-occurrence of all terms |
| 5 | software engineering requirements | Program-policy query with long-tail wording |
| 6 | database systems lab | Multi-term research intent, partial-match sensitivity |
| 7 | cybersecurity club meetings | Activity query; some pages mention only subset of terms |
| 8 | computer vision deep learning | Long topical query likely to fragment across pages |
| 9 | natural language processing course | Course-intent query with wording variation |
| 10 | distributed systems research | Broad technical query with many near-matches |
| 11 | human computer interaction | Phrase query that may split across tokenization variants |
| 12 | bioinformatics phd admissions | Specific academic-intent query with sparse exact overlap |

## B) Initially Good Queries (12)

| # | Query | Why this was selected as "good" initially |
|---|---|---|
| 1 | cristina lopes | Strong named-entity query |
| 2 | machine learning | Common two-term intent with high corpus support |
| 3 | ACM | Specific acronym/entity query |
| 4 | master of software engineering | Program name with strong exact matches |
| 5 | ics helpdesk | Specific service/entity query |
| 6 | uci informatics | High-frequency institutional entity query |
| 7 | graduate admissions | Common administrative intent |
| 8 | faculty directory | Stable navigational query |
| 9 | undergrad forms | Stable administrative query |
| 10 | tutoring center | Specific service query |
| 11 | honors program | Program-name style query |
| 12 | research labs | Broad but common institutional query |

All implementation changes, rationale, runtime notes, and demo commands are documented in the main report. Detailed results for all 24 queries are saved in milestone3_results.json.
