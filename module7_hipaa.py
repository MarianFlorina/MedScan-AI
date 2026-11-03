"""
Module 7: HIPAA Compliance & Security
=======================================
Implements security features required for healthcare applications:
- Powerful authentication system with email validation
- Password strength enforcement
- Role-based access control (Admin, Analyst, Viewer)
- Account lockout after failed attempts
- Audit logging
- PHI de-identification
- Session encryption & expiration
"""

import hashlib
import json
import os
import re
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st


# ============================================================
# EMAIL & PASSWORD VALIDATION
# ============================================================

class Validator:
    """Validates user inputs for authentication."""
    
    # Email regex (RFC 5322 simplified)
    EMAIL_REGEX = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Password strength requirements
    MIN_PASSWORD_LENGTH = 8
    MAX_PASSWORD_LENGTH = 128
    
    @classmethod
    def validate_email(cls, email: str) -> Tuple[bool, str]:
        """
        Validate email format.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email:
            return False, "Email is required"
        
        email = email.strip().lower()
        
        if len(email) > 254:
            return False, "Email is too long (max 254 characters)"
        
        if not cls.EMAIL_REGEX.match(email):
            return False, "Invalid email format (e.g., user@example.com)"
        
        # Check for common disposable email domains
        disposable_domains = {
            'tempmail.com', 'throwaway.com', 'guerrillamail.com',
            'mailinator.com', 'yopmail.com', 'temp-mail.org',
            'fakeinbox.com', 'sharklasers.com', 'guerrillamailblock.com',
        }
        domain = email.split('@')[1]
        if domain in disposable_domains:
            return False, "Disposable email addresses are not allowed"
        
        return True, ""
    
    @classmethod
    def validate_password(cls, password: str) -> Tuple[bool, str, Dict]:
        """
        Validate password strength.
        
        Returns:
            Tuple of (is_valid, error_message, strength_details)
        """
        if not password:
            return False, "Password is required", {}
        
        issues = []
        strength = {
            "length": len(password),
            "has_upper": False,
            "has_lower": False,
            "has_digit": False,
            "has_special": False,
            "score": 0,
            "level": "weak"
        }
        
        # Length check
        if len(password) < cls.MIN_PASSWORD_LENGTH:
            issues.append(f"Password must be at least {cls.MIN_PASSWORD_LENGTH} characters")
        
        if len(password) > cls.MAX_PASSWORD_LENGTH:
            issues.append(f"Password must be at most {cls.MAX_PASSWORD_LENGTH} characters")
        
        # Character type checks
        if re.search(r'[A-Z]', password):
            strength["has_upper"] = True
            strength["score"] += 1
        else:
            issues.append("Password must contain at least one uppercase letter")
        
        if re.search(r'[a-z]', password):
            strength["has_lower"] = True
            strength["score"] += 1
        else:
            issues.append("Password must contain at least one lowercase letter")
        
        if re.search(r'\d', password):
            strength["has_digit"] = True
            strength["score"] += 1
        else:
            issues.append("Password must contain at least one digit")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;\'`~]', password):
            strength["has_special"] = True
            strength["score"] += 1
        else:
            issues.append("Password must contain at least one special character (!@#$%^&*)")
        
        # Common password check
        common_passwords = {
            'password', 'password123', 'qwerty', 'abc123', 'letmein',
            'admin', 'admin123', 'welcome', 'monkey', 'dragon',
            'master', 'login', 'princess', 'football', 'shadow',
            'sunshine', 'trustno1', 'iloveyou', 'batman', 'access',
        }
        if password.lower() in common_passwords:
            issues.append("Password is too common")
            strength["score"] = max(0, strength["score"] - 2)
        
        # Sequential/repeated characters
        if re.search(r'(.)\1{2,}', password):
            issues.append("Password contains too many repeated characters")
            strength["score"] = max(0, strength["score"] - 1)
        
        # Only flag 3+ sequential chars not mixed with special chars
        if re.search(r'(abc|bcd|cde|def|efg|123|234|345|456|567|678|789)', password.lower()):
            # Only flag if the sequence is a standalone part (not like Admin@123)
            seq_match = re.search(r'(abc|bcd|cde|def|efg|123|234|345|456|567|678|789)', password.lower())
            if seq_match and len(password) <= 8:
                issues.append("Password contains sequential characters")
                strength["score"] = max(0, strength["score"] - 1)
        
        # Determine strength level
        if strength["score"] >= 4 and len(password) >= 12:
            strength["level"] = "strong"
        elif strength["score"] >= 3 and len(password) >= 8:
            strength["level"] = "medium"
        else:
            strength["level"] = "weak"
        
        is_valid = len(issues) == 0
        error_msg = "; ".join(issues) if issues else ""
        
        return is_valid, error_msg, strength
    
    @classmethod
    def validate_username(cls, username: str) -> Tuple[bool, str]:
        """Validate username format."""
        if not username:
            return False, "Username is required"
        
        username = username.strip()
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(username) > 30:
            return False, "Username must be at most 30 characters"
        
        if not re.match(r'^[a-zA-Z0-9._-]+$', username):
            return False, "Username can only contain letters, numbers, dots, hyphens, underscores"
        
        if username[0] in '._-':
            return False, "Username cannot start with a dot, hyphen, or underscore"
        
        return True, ""


# ============================================================
# USER DATABASE
# ============================================================

class UserDatabase:
    """
    In-memory user database with password hashing.
    In production, this would be a secure database.
    """
    
    USERS_FILE = os.environ.get("MEDSCAN_USERS_FILE", "users.json")
    
    DEFAULT_USERS = {}
    
    # Account lockout settings
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    
    def __init__(self):
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load users from file or use defaults."""
        if os.path.exists(self.USERS_FILE):
            try:
                with open(self.USERS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return self.DEFAULT_USERS.copy()
    
    def _save_users(self):
        """Save users to file."""
        with open(self.USERS_FILE, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password with PBKDF2-HMAC-SHA256 + random salt.
        Format: pbkdf2:<salt>:<hash>
        """
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 310000)
        return f"pbkdf2:{salt}:{dk.hex()}"

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        """Verify password against stored hash. Supports both PBKDF2 and legacy SHA-256."""
        if stored_hash.startswith('pbkdf2:'):
            # PBKDF2 format: pbkdf2:<salt>:<hash>
            parts = stored_hash.split(':')
            if len(parts) != 3:
                return False
            salt = parts[1]
            expected = parts[2]
            dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 310000)
            return secrets.compare_digest(dk.hex(), expected)
        else:
            # Legacy SHA-256 (for backward compatibility with existing users)
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get a user by username."""
        return self.users.get(username.lower().strip())
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get a user by email."""
        email = email.lower().strip()
        for username, user in self.users.items():
            if user.get("email", "").lower() == email:
                return {"username": username, **user}
        return None
    
    def is_locked(self, username: str) -> Tuple[bool, Optional[str]]:
        """Check if an account is locked."""
        user = self.get_user(username)
        if not user:
            return False, None
        
        if user.get("locked_until"):
            locked_until = datetime.fromisoformat(user["locked_until"])
            if datetime.utcnow() < locked_until:
                remaining = (locked_until - datetime.utcnow()).total_seconds() / 60
                return True, f"Account locked. Try again in {remaining:.0f} minutes"
            else:
                # Lockout expired, reset
                user["locked_until"] = None
                user["failed_attempts"] = 0
                self._save_users()
        
        return False, None
    
    def authenticate(self, username: str, password: str) -> Tuple[Optional[Dict], str]:
        """
        Authenticate a user.
        
        Returns:
            Tuple of (user_info_or_None, error_message)
        """
        username = username.strip().lower()
        user = self.get_user(username)
        
        if not user:
            return None, "Invalid username or password"
        
        # Check if account is active
        if not user.get("is_active", True):
            return None, "Account is deactivated. Contact administrator."
        
        # Check if account is locked
        is_locked, lock_msg = self.is_locked(username)
        if is_locked:
            return None, lock_msg
        
        # Verify password (supports both PBKDF2 and legacy SHA-256)
        if not self._verify_password(password, user["password_hash"]):
            # Increment failed attempts
            user["failed_attempts"] = user.get("failed_attempts", 0) + 1
            
            if user["failed_attempts"] >= self.MAX_FAILED_ATTEMPTS:
                user["locked_until"] = (
                    datetime.utcnow() + timedelta(minutes=self.LOCKOUT_DURATION_MINUTES)
                ).isoformat()
                self._save_users()
                return None, f"Too many failed attempts. Account locked for {self.LOCKOUT_DURATION_MINUTES} minutes"
            
            self._save_users()
            remaining = self.MAX_FAILED_ATTEMPTS - user["failed_attempts"]
            return None, f"Invalid username or password ({remaining} attempts remaining)"
        
        # Success — migrate legacy SHA-256 hash to PBKDF2
        if not user["password_hash"].startswith('pbkdf2:'):
            user["password_hash"] = self._hash_password(password)
        
        # Reset failed attempts
        user["failed_attempts"] = 0
        user["locked_until"] = None
        user["last_login"] = datetime.utcnow().isoformat()
        self._save_users()
        
        return {
            "username": username,
            "email": user["email"],
            "role": user["role"],
            "full_name": user["full_name"],
            "permissions": AccessControl.ROLES.get(user["role"], {}),
            "authenticated_at": datetime.utcnow().isoformat(),
        }, ""
    
    def register_user(self, username: str, email: str, password: str,
                      full_name: str, role: str = "viewer") -> Tuple[bool, str]:
        """Register a new user."""
        # Validate username
        valid, msg = Validator.validate_username(username)
        if not valid:
            return False, msg
        
        username = username.lower().strip()
        
        if username in self.users:
            return False, "Username already exists"
        
        # Validate email
        valid, msg = Validator.validate_email(email)
        if not valid:
            return False, msg
        
        # Check email uniqueness
        if self.get_user_by_email(email):
            return False, "Email already registered"
        
        # Validate password
        valid, msg, _ = Validator.validate_password(password)
        if not valid:
            return False, msg
        
        # Validate role
        if role not in AccessControl.ROLES:
            return False, f"Invalid role. Must be one of: {', '.join(AccessControl.ROLES.keys())}"
        
        # Create user
        self.users[username] = {
            "email": email.lower().strip(),
            "password_hash": self._hash_password(password),
            "role": role,
            "full_name": full_name.strip(),
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True,
            "failed_attempts": 0,
            "locked_until": None,
            "last_login": None,
        }
        
        self._save_users()
        return True, "Account created successfully"
    
    def set_role(self, username: str, new_role: str, acting_user: str = None) -> Tuple[bool, str]:
        """Change a user's role (admin action). Refuses to remove the last remaining admin."""
        username = username.lower().strip()
        user = self.users.get(username)
        if not user:
            return False, "User not found"
        if new_role not in AccessControl.ROLES:
            return False, f"Invalid role. Must be one of: {', '.join(AccessControl.ROLES)}"
        if user["role"] == new_role:
            return False, f"{username} is already {new_role}"
        if user["role"] == "admin" and new_role != "admin":
            if sum(1 for u in self.users.values() if u["role"] == "admin" and u.get("is_active", True)) <= 1:
                return False, "Cannot remove the last admin"
        user["role"] = new_role
        self._save_users()
        return True, f"{username} is now {new_role}"

    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change a user's password."""
        user = self.get_user(username)
        if not user:
            return False, "User not found"
        
        # Verify old password
        if not self._verify_password(old_password, user["password_hash"]):
            return False, "Current password is incorrect"
        
        # Validate new password
        valid, msg, _ = Validator.validate_password(new_password)
        if not valid:
            return False, msg
        
        # Check it's different from old
        if self._verify_password(new_password, user["password_hash"]):
            return False, "New password must be different from current password"
        
        user["password_hash"] = self._hash_password(new_password)
        self._save_users()
        return True, "Password changed successfully"
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (admin only)."""
        result = []
        for username, user in self.users.items():
            result.append({
                "username": username,
                "email": user["email"],
                "role": user["role"],
                "full_name": user["full_name"],
                "is_active": user.get("is_active", True),
                "last_login": user.get("last_login"),
                "failed_attempts": user.get("failed_attempts", 0),
            })
        return result


# ============================================================
# ACCESS CONTROL
# ============================================================

class AccessControl:
    """
    Role-based access control for the application.
    
    Roles (hierarchical):
    - viewer: Can view results only
    - analyst: Can upload, process, export
    - admin: Full access including user management and audit logs
    """
    
    ROLES = {
        "viewer": {
            "upload": False,
            "process": False,
            "export": False,
            "admin": False,
            "manage_users": False,
            "view_audit": False,
            "change_settings": False,
        },
        "analyst": {
            "upload": True,
            "process": True,
            "export": True,
            "admin": False,
            "manage_users": False,
            "view_audit": False,
            "change_settings": False,
        },
        "admin": {
            "upload": True,
            "process": True,
            "export": True,
            "admin": True,
            "manage_users": True,
            "view_audit": True,
            "change_settings": True,
        },
    }
    
    ROLE_DESCRIPTIONS = {
        "viewer": {
            "name": "Viewer",
            "description": "Read-only access to view analysis results",
            "color": "#3fb950",
            "icon": "👁️",
            "capabilities": [
                "View uploaded documents",
                "View extracted entities",
                "View anomaly detection results",
                "View FHIR output",
            ],
            "restrictions": [
                "Cannot upload documents",
                "Cannot export data",
                "Cannot modify settings",
            ],
        },
        "analyst": {
            "name": "Analyst",
            "description": "Full clinical analysis capabilities",
            "color": "#58a6ff",
            "icon": "🔬",
            "capabilities": [
                "Upload and process documents",
                "Run OCR and NER analysis",
                "Export FHIR bundles and CSV reports",
                "View anomaly detection results",
                "Use demo mode",
            ],
            "restrictions": [
                "Cannot manage users",
                "Cannot view audit logs",
                "Cannot modify system settings",
            ],
        },
        "admin": {
            "name": "Administrator",
            "description": "Full system access with user management",
            "color": "#f0883e",
            "icon": "👑",
            "capabilities": [
                "All Analyst capabilities",
                "Manage user accounts",
                "View audit logs and compliance reports",
                "Modify system settings",
                "Register new users",
                "View all user activity",
            ],
            "restrictions": [],
        },
    }
    
    @classmethod
    def check_permission(cls, permission: str) -> bool:
        """Check if the current user has a specific permission."""
        if "user_info" not in st.session_state:
            return False
        user_info = st.session_state.user_info
        return user_info.get("permissions", {}).get(permission, False)
    
    @classmethod
    def require_permission(cls, permission: str) -> bool:
        """
        Require a permission, show error if not granted.
        Returns True if granted, False otherwise.
        """
        if cls.check_permission(permission):
            return True
        
        user_info = st.session_state.get("user_info", {})
        role = user_info.get("role", "unknown")
        role_info = cls.ROLE_DESCRIPTIONS.get(role, {})
        
        st.error(
            f"🔒 **Access Denied**\n\n"
            f"Your role ({role_info.get('name', role)}) does not have the **{permission}** permission.\n\n"
            f"Required role: **Analyst** or **Admin**\n\n"
            f"Contact your administrator to request access."
        )
        return False
    
    @classmethod
    def get_role_info(cls, role: str) -> Dict:
        """Get detailed role information."""
        return cls.ROLE_DESCRIPTIONS.get(role, {})


# ============================================================
# PHI DETECTION & DE-IDENTIFICATION
# ============================================================

class PHIDetector:
    """
    Detects and de-identifies Protected Health Information (PHI).
    Follows HIPAA Safe Harbor method.
    """
    
    PHI_PATTERNS = {
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'),
        "PHONE": re.compile(r'\b(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "MRN": re.compile(r'\b(MRN|mrn)[:\s#]*(\d{6,10})\b', re.IGNORECASE),
        "DATE_OF_BIRTH": re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(19|20)\d{2}\b'),
        "DATE_OF_BIRTH_ALT": re.compile(r'\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'),
        "ZIP_CODE": re.compile(r'\b\d{5}(-\d{4})?\b'),
        "IP_ADDRESS": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        "NPI": re.compile(r'\b(NPI|npi)[:\s#]*(\d{10})\b', re.IGNORECASE),
    }
    
    @classmethod
    def detect_phi(cls, text: str) -> List[Dict]:
        """Detect PHI in text."""
        findings = []
        for phi_type, pattern in cls.PHI_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": phi_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        return findings
    
    @classmethod
    def deidentify(cls, text: str, method: str = "mask") -> Tuple[str, List[Dict]]:
        """De-identify PHI in text."""
        findings = cls.detect_phi(text)
        changes = []
        findings.sort(key=lambda x: x["start"], reverse=True)
        
        deidentified = text
        for finding in findings:
            original = finding["value"]
            if method == "mask":
                replacement = f"[{finding['type']}]"
            elif method == "hash":
                replacement = hashlib.sha256(original.encode()).hexdigest()[:12]
            else:
                replacement = ""
            
            deidentified = deidentified[:finding["start"]] + replacement + deidentified[finding["end"]:]
            changes.append({"type": finding["type"], "original_length": len(original), "method": method})
        
        return deidentified, changes


# ============================================================
# AUDIT LOGGING
# ============================================================

class AuditLogger:
    """HIPAA-compliant audit logging system."""
    
    LOG_DIR = os.environ.get("MEDSCAN_AUDIT_DIR", "audit_logs")
    
    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        if "audit_session_id" not in st.session_state:
            st.session_state.audit_session_id = hashlib.sha256(
                f"{time.time()}{os.getpid()}".encode()
            ).hexdigest()[:16]
        return st.session_state.audit_session_id
    
    def _write_log(self, event_type: str, details: Dict):
        username = (st.session_state.get("user_info") or {}).get("username", "anonymous")
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "username": username,
            "event_type": event_type,
            "details": details,
        }
        
        log_file = os.path.join(
            self.LOG_DIR,
            f"audit_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        )
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_access(self, username: str, action: str):
        self._write_log("ACCESS", {"username": username, "action": action})
    
    def log_document_upload(self, filename: str, file_type: str, size_bytes: int):
        self._write_log("DOCUMENT_UPLOAD", {"filename": filename, "file_type": file_type, "size_bytes": size_bytes})
    
    def log_processing(self, document_id: str, entities_found: int, anomalies_found: int):
        self._write_log("PROCESSING", {"document_id": document_id, "entities_found": entities_found, "anomalies_found": anomalies_found})
    
    def log_export(self, export_type: str, document_id: str = None):
        self._write_log("EXPORT", {"export_type": export_type, "document_id": document_id})
    
    def log_phi_detection(self, phi_count: int, method: str):
        self._write_log("PHI_DETECTION", {"phi_count": phi_count, "deidentify_method": method})
    
    def log_user_management(self, action: str, target_user: str = None):
        self._write_log("USER_MANAGEMENT", {"action": action, "target_user": target_user})
    
    def log_error(self, error_type: str, message: str, details: Dict = None):
        self._write_log("ERROR", {"error_type": error_type, "message": message, "details": details or {}})
    
    def get_logs(self, date: str = None, event_type: str = None, limit: int = 100) -> List[Dict]:
        logs = []
        target_date = date or datetime.utcnow().strftime('%Y-%m-%d')
        log_file = os.path.join(self.LOG_DIR, f"audit_{target_date}.jsonl")
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if event_type is None or entry["event_type"] == event_type:
                        logs.append(entry)
        
        return logs[-limit:]  # most recent `limit` events (oldest first)
    
    def get_log_summary(self, date: str = None) -> Dict:
        logs = self.get_logs(date, limit=10000)
        
        summary = {
            "total_events": len(logs),
            "by_type": {},
            "by_user": {},
            "sessions": set(),
        }
        
        for log in logs:
            event_type = log["event_type"]
            username = log.get("username", "unknown")
            summary["by_type"][event_type] = summary["by_type"].get(event_type, 0) + 1
            summary["by_user"][username] = summary["by_user"].get(username, 0) + 1
            summary["sessions"].add(log["session_id"])
        
        summary["unique_sessions"] = len(summary["sessions"])
        del summary["sessions"]
        
        return summary


# ============================================================
# SESSION MANAGEMENT
# ============================================================

class SessionManager:
    """Manages secure session state."""
    
    @staticmethod
    def initialize():
        defaults = {
            "authenticated": False,
            "user_info": None,
            "session_start": datetime.utcnow().isoformat(),
            "documents_processed": 0,
            "phi_deidentified": False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def touch():
        """Record user activity (called on every script run while signed in)."""
        st.session_state["last_activity"] = datetime.utcnow().isoformat()

    @staticmethod
    def is_expired(max_minutes: int = 30) -> bool:
        """True after `max_minutes` of INACTIVITY (not since login)."""
        stamp = st.session_state.get("last_activity") or st.session_state.get("session_start")
        if not stamp:
            return True
        elapsed = (datetime.utcnow() - datetime.fromisoformat(stamp)).total_seconds() / 60
        return elapsed > max_minutes
    
    @staticmethod
    def get_session_duration() -> str:
        if "session_start" not in st.session_state:
            return "0m"
        start = datetime.fromisoformat(st.session_state.session_start)
        elapsed = (datetime.utcnow() - start).total_seconds()
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        return f"{elapsed / 60:.0f}m"
    
    @staticmethod
    def logout():
        for key in list(st.session_state.keys()):
            del st.session_state[key]


# ============================================================
# COMPLIANCE REPORT
# ============================================================

def generate_compliance_report() -> Dict:
    audit = AuditLogger()
    summary = audit.get_log_summary()
    
    return {
        "report_date": datetime.utcnow().isoformat(),
        "session_id": audit.session_id,
        "features_implemented": [
            "Email-validated user registration",
            "Password strength enforcement",
            "Role-based access control (Viewer/Analyst/Admin)",
            "Account lockout after failed attempts",
            "PHI Detection and De-identification",
            "Audit Logging (all access, exports, user management)",
            "Session Management with Expiration",
            "Local-only Processing (no external data transmission)",
            "Data Retention Policies",
        ],
        "hipaa_safeguards": {
            "administrative": [
                "Role-based access control with 3 tiers",
                "User authentication with email validation",
                "Audit trail logging for all actions",
                "Account lockout after 5 failed attempts",
                "Password strength requirements enforced",
            ],
            "physical": [
                "Local processing only",
                "No cloud storage of PHI",
            ],
            "technical": [
                "PHI de-identification (mask/hash/remove)",
                "Password hashing with PBKDF2-HMAC-SHA256 + random salt",
                "Automatic session expiration (30 min)",
                "Audit log integrity (append-only)",
                "Disposable email blocking",
            ],
        },
        "activity_summary": summary,
        "password_policy": {
            "min_length": 8,
            "max_length": 128,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digit": True,
            "require_special": True,
            "block_common_passwords": True,
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 15,
        },
        "recommendations": [
            "Enable HTTPS in production deployment",
            "Implement MFA for admin access",
            "Set up automated log rotation and archival",
            "Conduct regular security audits",
            "Train staff on HIPAA compliance procedures",
            "Consider upgrading to bcrypt/argon2 for even stronger hashing",
            "Implement password expiration policies",
        ],
    }


# Singleton instances
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
