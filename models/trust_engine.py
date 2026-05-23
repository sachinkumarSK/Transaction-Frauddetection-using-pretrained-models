"""
Dynamic Trust Engine
=====================
Continuously updates a per-customer trust score based on:
  - ML fraud probability (LightGBM)
  - Anomaly detection flags (Isolation Forest)
  - Session behaviour analytics
  - Device trust signals
  - Geo/location risk
  - Transaction velocity

The trust score DECAYS on suspicious activity and RECOVERS slowly
over normal behaviour — mimicking how real banks track customer risk.

Innovation: Traditional systems score TRANSACTIONS.
            This engine scores the CUSTOMER continuously.
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


@dataclass
class CustomerProfile:
    """Tracks continuous trust state for a customer."""
    customer_id: str
    trust_score: float = 85.0           # starts at trusted baseline
    baseline_trust: float = 85.0
    total_transactions: int = 0
    fraud_flags: int = 0
    mfa_challenges: int = 0
    last_transaction_time: float = 0.0
    last_device_hash: str = ""
    last_geo_lat: float = 0.0
    last_geo_lon: float = 0.0
    consecutive_normal: int = 0         # consecutive normal txns
    trust_history: list = field(default_factory=list)


class DynamicTrustEngine:
    """
    Maintains per-customer trust scores that evolve over time.
    
    Trust score range: 0 (fully untrusted) to 100 (fully trusted)
    
    Trust impacts decisions:
      trust >= 70  → faster approvals, lower friction
      trust 40-70  → additional verification (MFA)
      trust < 40   → block and flag for review
    """
    
    def __init__(self):
        self.customers: dict[str, CustomerProfile] = {}
        self.fraud_network: dict[str, float] = {}  # device/beneficiary reputation
    
    def get_or_create(self, customer_id: str) -> CustomerProfile:
        if customer_id not in self.customers:
            self.customers[customer_id] = CustomerProfile(customer_id=customer_id)
        return self.customers[customer_id]
    
    def compute_trust_adjustment(
        self,
        customer_id: str,
        ml_fraud_score: float,          # 0-1 from LightGBM
        anomaly_flag: bool,             # from Isolation Forest
        session_risk: float,            # 0-100 session risk score
        device_trust: float,            # 0-100 device trust
        geo_risk: float,               # 0-100 geo risk
        vpn_detected: bool,
        new_device: bool,
        impossible_travel: bool,
        txn_amount: float,
        avg_amount: float,
        failed_otp: int = 0,
        paste_detected: bool = False,
        beneficiary_new: bool = False,
    ) -> dict:
        """
        Calculate trust score adjustment and final decision confidence.
        
        Returns dict with:
          - trust_score: updated customer trust
          - trust_delta: how much trust changed
          - risk_multiplier: applied to ML score
          - decision_confidence: overall confidence in decision
          - trust_factors: breakdown of what affected trust
        """
        profile = self.get_or_create(customer_id)
        old_trust = profile.trust_score
        
        trust_factors = []
        adjustment = 0.0
        
        # ── ML Score Impact ──
        if ml_fraud_score > 0.8:
            adjustment -= 25
            trust_factors.append(("ML fraud score critical", -25))
        elif ml_fraud_score > 0.5:
            adjustment -= 15
            trust_factors.append(("ML fraud score elevated", -15))
        elif ml_fraud_score > 0.3:
            adjustment -= 5
            trust_factors.append(("ML fraud score moderate", -5))
        elif ml_fraud_score < 0.1:
            adjustment += 2
            trust_factors.append(("ML score clean", +2))
        
        # ── Anomaly Impact ──
        if anomaly_flag:
            adjustment -= 12
            trust_factors.append(("Anomaly detected (IsoForest)", -12))
        
        # ── Session Behaviour ──
        if session_risk > 60:
            adjustment -= 10
            trust_factors.append(("High session risk", -10))
        elif session_risk > 30:
            adjustment -= 3
            trust_factors.append(("Moderate session risk", -3))
        
        if paste_detected:
            adjustment -= 5
            trust_factors.append(("Paste detected (potential phishing)", -5))
        
        if failed_otp >= 2:
            adjustment -= 8 * failed_otp
            trust_factors.append((f"Failed OTP x{failed_otp}", -8 * failed_otp))
        
        # ── Device Trust ──
        if new_device:
            adjustment -= 8
            trust_factors.append(("New/unknown device", -8))
        
        if device_trust < 30:
            adjustment -= 10
            trust_factors.append(("Very low device trust", -10))
        elif device_trust > 80:
            adjustment += 3
            trust_factors.append(("Trusted device", +3))
        
        # ── Geo Risk ──
        if impossible_travel:
            adjustment -= 20
            trust_factors.append(("Impossible travel detected", -20))
        
        if vpn_detected:
            adjustment -= 7
            trust_factors.append(("VPN/proxy detected", -7))
        
        if geo_risk > 60:
            adjustment -= 5
            trust_factors.append(("High-risk geography", -5))
        
        # ── Amount Anomaly ──
        amount_ratio = txn_amount / max(avg_amount, 1)
        if amount_ratio > 10:
            adjustment -= 15
            trust_factors.append((f"Amount {amount_ratio:.1f}x above baseline", -15))
        elif amount_ratio > 5:
            adjustment -= 8
            trust_factors.append((f"Amount {amount_ratio:.1f}x above baseline", -8))
        
        # ── Beneficiary Risk ──
        if beneficiary_new:
            adjustment -= 5
            trust_factors.append(("New beneficiary added recently", -5))
        
        # ── Consecutive Normal Behaviour Recovery ──
        if adjustment >= 0:
            profile.consecutive_normal += 1
            # Slow recovery: +1 per normal transaction, capped
            recovery = min(profile.consecutive_normal * 0.5, 3)
            adjustment += recovery
            if recovery > 0:
                trust_factors.append((f"Trust recovery (consecutive normal: {profile.consecutive_normal})", round(recovery, 1)))
        else:
            profile.consecutive_normal = 0
        
        # ── Time Decay ──
        # Trust naturally recovers slightly over time (12 hours = +1 point)
        if profile.last_transaction_time > 0:
            hours_since = (time.time() - profile.last_transaction_time) / 3600
            time_recovery = min(hours_since / 12, 2)  # max +2 from time decay
            adjustment += time_recovery
        
        # ── Apply Adjustment ──
        new_trust = max(0, min(100, old_trust + adjustment))
        profile.trust_score = new_trust
        profile.total_transactions += 1
        profile.last_transaction_time = time.time()
        
        if ml_fraud_score > 0.5:
            profile.fraud_flags += 1
        
        # Track history
        profile.trust_history.append({
            "trust": round(new_trust, 1),
            "delta": round(new_trust - old_trust, 1),
            "timestamp": time.time()
        })
        if len(profile.trust_history) > 100:
            profile.trust_history.pop(0)
        
        # ── Risk Multiplier ──
        # Low trust amplifies risk; high trust dampens it
        if new_trust < 30:
            risk_multiplier = 1.4
        elif new_trust < 50:
            risk_multiplier = 1.2
        elif new_trust > 80:
            risk_multiplier = 0.85
        else:
            risk_multiplier = 1.0
        
        # ── Decision Confidence ──
        # How confident are we in the final decision?
        signal_count = sum(1 for _, adj in trust_factors if adj < 0)
        if signal_count >= 4:
            decision_confidence = 0.95
        elif signal_count >= 2:
            decision_confidence = 0.80
        elif signal_count >= 1:
            decision_confidence = 0.65
        else:
            decision_confidence = 0.50
        
        return {
            "trust_score": round(new_trust, 1),
            "trust_delta": round(new_trust - old_trust, 1),
            "previous_trust": round(old_trust, 1),
            "risk_multiplier": risk_multiplier,
            "decision_confidence": decision_confidence,
            "trust_factors": [(f, round(v, 1)) for f, v in trust_factors],
            "consecutive_normal": profile.consecutive_normal,
            "total_transactions": profile.total_transactions,
            "fraud_flags": profile.fraud_flags,
        }
    
    def get_customer_summary(self, customer_id: str) -> Optional[dict]:
        if customer_id not in self.customers:
            return None
        p = self.customers[customer_id]
        return {
            "customer_id": p.customer_id,
            "trust_score": round(p.trust_score, 1),
            "total_transactions": p.total_transactions,
            "fraud_flags": p.fraud_flags,
            "consecutive_normal": p.consecutive_normal,
            "trust_history": p.trust_history[-20:],
        }
    
    def update_network_reputation(self, entity_id: str, risk_delta: float):
        """Update fraud network reputation for device/beneficiary."""
        current = self.fraud_network.get(entity_id, 50.0)
        self.fraud_network[entity_id] = max(0, min(100, current + risk_delta))
    
    def get_network_risk(self, entity_id: str) -> float:
        return self.fraud_network.get(entity_id, 50.0)
