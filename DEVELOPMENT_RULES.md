# DEVELOPMENT RULES - PREVENTING CODE BREAKAGE

## 🚨 CRITICAL RULE: NEVER BREAK EXISTING WORKING SCANNERS

### **PROBLEM PATTERN IDENTIFIED:**
Every time we integrate new scanners, existing working scanners break due to:
1. **Shared infrastructure changes** affecting multiple scanners
2. **Database logger modifications** breaking existing scanner connections
3. **Environment variable changes** breaking cron job execution
4. **Import path modifications** causing import errors
5. **Schema changes** affecting existing data flows

### **MANDATORY PRE-INTEGRATION CHECKLIST:**

#### **1. BASELINE VERIFICATION (BEFORE ANY CHANGES):**
- [ ] **Test ALL existing scanners manually** to confirm they're working
- [ ] **Verify ALL dashboards** are displaying data correctly
- [ ] **Check ALL cron jobs** are executing and creating log files
- [ ] **Document current working state** with screenshots/logs
- [ ] **Create backup** of current working configuration

#### **2. ISOLATION STRATEGY:**
- [ ] **New scanners MUST use separate database schemas** when possible
- [ ] **New scanners MUST use separate loggers** when possible
- [ ] **New scanners MUST NOT modify** existing scanner files
- [ ] **New scanners MUST NOT change** existing environment variables
- [ ] **New scanners MUST NOT modify** existing import paths

#### **3. INTEGRATION APPROACH:**
- [ ] **Create new files** instead of modifying existing ones
- [ ] **Use inheritance/composition** instead of direct modification
- [ ] **Test new functionality** in isolation before integration
- [ ] **Verify existing functionality** after each change
- [ ] **Rollback immediately** if any existing scanner breaks

#### **4. TESTING REQUIREMENTS:**
- [ ] **Run ALL existing scanners** after each change
- [ ] **Verify ALL dashboards** still display data
- [ ] **Check ALL cron jobs** still execute
- [ ] **Confirm ALL database connections** still work
- [ ] **Validate ALL data flows** remain intact

### **SPECIFIC RULES FOR SCANNER INTEGRATION:**

#### **Database Logging:**
- ✅ **DO:** Create new logger classes for new scanners
- ✅ **DO:** Use separate database schemas when possible
- ✅ **DO:** Test database connections before deployment
- ❌ **DON'T:** Modify existing `TradeLogger` or `WyckoffLogger` classes
- ❌ **DON'T:** Change existing database connection strings
- ❌ **DON'T:** Modify existing table structures

#### **Environment Variables:**
- ✅ **DO:** Use explicit `export DATABASE_URL` in cron jobs
- ✅ **DO:** Test environment variable loading in cron context
- ✅ **DO:** Verify `.env` file compatibility
- ❌ **DON'T:** Rely on `source .env` in cron jobs
- ❌ **DON'T:** Change existing environment variable names
- ❌ **DON'T:** Remove existing environment variables

#### **Import Paths:**
- ✅ **DO:** Use absolute paths or proper relative paths
- ✅ **DO:** Test imports in target environment
- ✅ **DO:** Verify file locations before import
- ❌ **DON'T:** Use hardcoded paths like `'modules'`
- ❌ **DON'T:** Assume file locations without verification
- ❌ **DON'T:** Break existing import chains

#### **Cron Jobs:**
- ✅ **DO:** Test cron job execution manually first
- ✅ **DO:** Verify log file creation and content
- ✅ **DO:** Use explicit environment variable exports
- ❌ **DON'T:** Deploy untested cron configurations
- ❌ **DON'T:** Rely on environment file sourcing in cron
- ❌ **DON'T:** Assume cron will work without testing

### **BREAKAGE PREVENTION WORKFLOW:**

1. **BEFORE STARTING:**
   - Document current working state
   - Test all existing functionality
   - Create backup/checkpoint

2. **DURING DEVELOPMENT:**
   - Make minimal, targeted changes
   - Test after each change
   - Verify existing functionality intact

3. **BEFORE DEPLOYMENT:**
   - Run comprehensive testing
   - Verify all scanners work
   - Confirm all dashboards functional

4. **AFTER DEPLOYMENT:**
   - Monitor for any issues
   - Test all functionality again
   - Rollback immediately if problems

### **EMERGENCY ROLLBACK PROCEDURE:**
If any existing scanner breaks:
1. **STOP** all development immediately
2. **REVERT** to last working commit
3. **RESTORE** from backup if needed
4. **VERIFY** all existing functionality restored
5. **ANALYZE** what caused the breakage
6. **DOCUMENT** the failure for future prevention

### **SUCCESS METRICS:**
- ✅ **ALL existing scanners** continue working
- ✅ **ALL existing dashboards** continue displaying data
- ✅ **ALL existing cron jobs** continue executing
- ✅ **ALL existing database connections** remain functional
- ✅ **NEW functionality** works as intended
- ✅ **ZERO regression** in existing features

### **ACCOUNTABILITY:**
- **Every breakage** must be documented
- **Root cause analysis** must be performed
- **Prevention measures** must be implemented
- **Testing procedures** must be enhanced
- **Rollback procedures** must be practiced

---

**REMEMBER: It's better to take longer to integrate new features than to break existing working functionality. The cost of fixing broken systems far exceeds the time saved by rushing integration.**
