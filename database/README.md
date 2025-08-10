# Database Module

This directory contains database-related files for the BB Scanner.

## 📊 Files

### `create_schema.sql`
- **Purpose:** Complete database schema creation script
- **Usage:** Run to create the full database structure
- **Description:** Creates all tables, indexes, and constraints for the BB Scanner database

### `create_schema_simple.sql`
- **Purpose:** Simplified database schema creation script
- **Usage:** Run for basic database setup
- **Description:** Creates essential tables for basic functionality

### `verify_comprehensive_data.py`
- **Purpose:** Verify comprehensive data capture in database
- **Usage:** `python verify_comprehensive_data.py`
- **Description:** Checks that all 139+ Excel fields are properly captured in the database scanner_specific_data JSON

## 🚀 Setup

To set up the database:

```bash
# For complete setup
psql -d your_database -f create_schema.sql

# For simple setup
psql -d your_database -f create_schema_simple.sql
```

## ✅ Verification

To verify data capture:

```bash
python verify_comprehensive_data.py
```
