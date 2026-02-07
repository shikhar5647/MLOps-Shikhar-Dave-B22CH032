# Assignment 2: Data Contracts for ML & Data Systems

This repository branch contains **data contracts** implemented as YAML files for four real-world scenarios involving Machine Learning and data pipelines.  
Each contract formalizes the agreement between **data producers** and **data consumers**, ensuring schema stability, data quality, governance, and reliable downstream ML behavior.

---

## What is a Data Contract?

A **data contract** is a formal, machine-readable agreement between a **producer** and a **consumer** of data.  
It defines:

- Logical schema (business-friendly field names)
- Data quality rules (validity, completeness, regex, ranges)
- Enforcement behavior (soft vs hard / circuit breakers)
- SLAs (freshness, availability, retention)
- Governance (PII tagging)
- Lineage and ownership

In modern ML systems, data contracts act as the **API layer for data**, preventing silent failures caused by schema drift, bad values, or upstream changes.

---

## Why Data Contracts Matter for ML

- Prevent **silent model failures**
- Catch **bad data early** (before training or inference)
- Decouple consumers from fragile physical schemas
- Make enforcement explicit (fail fast vs warn)
- Improve trust, observability, and compliance

---

## Repository Structure

```
.
├── rides_contract.yaml       # Scenario 1: Ride-share dynamic pricing
├── orders_contract.yaml      # Scenario 2: E-commerce flash-sale orders
├── thermostat_contract.yaml  # Scenario 3: IoT thermostat telemetry
├── fintech_contract.yaml     # Scenario 4: FinTech transaction log
└── README.md
```

---

## Scenario Overview

### **Scenario 1: Ride-Share Events (Dynamic Pricing)**
**File:** `rides_contract.yaml`

**Context:**  
A dynamic pricing ML model depends on clean ride event data.

**Key Contract Rules:**
- Fare must be non-negative (HARD)
- Driver rating must be between 1.0 and 5.0 (HARD)
- Distance must be present (HARD)
- Passenger ID marked as PII (soft governance)

**Why:**  
Invalid values can crash pricing models or produce incorrect fares.

---

### **Scenario 2: Flash Sale Orders (E-commerce)**
**File:** `orders_contract.yaml`

**Context:**  
A real-time marketing dashboard consumes order data during flash sales.

**Key Contract Rules:**
- Order total must be non-negative (HARD)
- Order status must map to a controlled enum (HARD)
- Unmapped status codes are rejected (HARD)

**Why:**  
Unexpected status codes previously caused dashboard crashes during peak traffic.

---

### **Scenario 3: Smart Thermostat Telemetry (IoT)**
**File:** `thermostat_contract.yaml`

**Context:**  
IoT devices send minute-level temperature and battery telemetry.

**Key Contract Rules:**
- Temperature must be in range −30°C to 60°C (HARD)
- Sentinel value `9999` explicitly rejected (HARD)
- Battery level must be between 0.0 and 1.0 (HARD)

**Why:**  
Outliers and sentinel values distort aggregate analytics.

**Enforcement Strategy:**  
Invalid records are quarantined instead of blocking the entire stream.

---

### **Scenario 4: FinTech Transaction Log (Fraud Detection)**
**File:** `fintech_contract.yaml`

**Context:**  
A fraud detection system performs real-time account lookups.

**Problem:**  
Legacy upgrades produced malformed source account IDs, causing silent lookup failures.

**Key Contract Rules:**
- `acct_src` must match regex `^[A-Z0-9]{10}$` (HARD CIRCUIT BREAKER)
- Any violation blocks the pipeline immediately

**Why:**  
Malformed account IDs cause fraud models to miss suspicious transactions.

---

## Enforcement Philosophy

| Rule Type | Enforcement | Example |
|-----------|-------------|---------|
| Model-critical | HARD (block pipeline) | Negative fares, invalid account IDs |
| Analytical quality | HARD or quarantine | Sensor outliers |
| Governance (PII) | SOFT (mask + warn) | Passenger ID |

A **Hard Circuit Breaker** means:
- Pipeline is blocked
- Bad data is quarantined
- Teams are notified immediately
- No silent fallback or auto-fix

---

## Validation & Linting

### YAML validation
```bash
pip install yamllint
yamllint *.yaml
```

### Regex example (Scenario 4)
```
^[A-Z0-9]{10}$
```

**Valid:**
- `ABCD123456`
- `A1B2C3D4E5`
- `1234567890`

**Invalid:**
- `abcd1234` (lowercase letters)
- `ABC_123456` (underscore not allowed)
- `ABC123` (too short)

---

## Where Contracts Are Enforced

Typical enforcement points:

- Kafka Connect / streaming validators
- Pre-ingestion validation services
- Airflow or batch pre-flight checks
- Schema registry + custom rule engine

Contracts are evaluated before data reaches:

- ML training
- ML inference
- Dashboards
- Analytics pipelines

---

## Key Learnings from This Assignment

- Data contracts are first-class ML infrastructure
- Validation must be explicit, not implicit
- Hard vs soft enforcement is a business decision
- Regex and pattern rules are critical for joins and lookups
- Good contracts prevent silent failures better than monitoring alone

---

## Conclusion

This assignment demonstrates how data contracts can be used to:

- Protect ML systems from bad data
- Formalize producer–consumer agreements
- Enforce quality, governance, and reliability
- Enable scalable and maintainable data platforms

Each scenario highlights a different failure mode and shows how a well-designed data contract prevents it.

---

## Author : Shikhar Dave
