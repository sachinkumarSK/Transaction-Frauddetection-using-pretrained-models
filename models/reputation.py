"""
Reputation Store  (Beneficiary + Device Reputation Service — PRD §7 step 3, §10)
================================================================================
A lightweight, in-memory stand-in for the permissioned-blockchain reputation
ledger described in the PRD. It holds a risk reputation (0 = clean, 100 =
known-bad) for two kinds of entity:

  - beneficiary  (the receiving account / UPI VPA)
  - device       (a device fingerprint hash)

The PRD's blockchain stores exactly this: beneficiary reputation, device
reputation and mule-account intelligence — "No customer personal information
or transaction payloads are stored on-chain." We keep only opaque IDs + risk.

Phase 2 (this file)  : seeded known-bad entities + lookup feeding the TTI.
Phase 3 (feedback loop): report_fraud() is called when a txn is confirmed
                         fraud, so the NEXT transfer to the same beneficiary /
                         device is scored higher automatically.
"""

import time
from typing import Optional


class ReputationStore:
    """In-memory reputation ledger for beneficiaries and devices."""

    # Reputation of an entity we have never seen. A brand-new beneficiary is not
    # "bad", but it has no track record — the TTI applies a first-time premium.
    UNKNOWN_RISK = 0.0

    def __init__(self, db=None):
        # entity_id -> record dict
        self.entities: dict[str, dict] = {}
        self.db = db
        loaded = db.load_reputation() if db else []
        if loaded:
            for r in loaded:
                self.entities[r["entity_id"]] = dict(r)
        else:
            self._seed()
            if db:
                for rec in self.entities.values():
                    db.upsert_reputation(rec)

    def _persist(self, rec):
        if self.db:
            self.db.upsert_reputation(rec)

    def _seed(self):
        """Seed with entities flagged by prior investigations / consortium intel.

        Represents fraud fingerprints already committed to the shared ledger
        before this session started.
        """
        seed = [
            # (entity_id,        kind,          risk, fraud_reports, note)
            ("BENF-MULE-001",   "beneficiary", 92.0, 7, "Known mule — layering ring"),
            ("BENF-MULE-002",   "beneficiary", 85.0, 4, "Mule account, multiple STRs"),
            ("BENF-SCAM-088",   "beneficiary", 78.0, 3, "APP fraud payout account"),
            ("DEV-BOTNET-0x14", "device",      88.0, 5, "Emulator farm fingerprint"),
            ("DEV-ATO-0x77",    "device",      74.0, 2, "Account-takeover device"),
        ]
        now = time.time()
        for eid, kind, risk, reports, note in seed:
            self.entities[eid] = {
                "entity_id": eid, "kind": kind, "risk": risk,
                "fraud_reports": reports, "times_seen": reports,
                "first_seen": now, "last_updated": now, "note": note,
                "source": "consortium_ledger",
            }

    def lookup(self, entity_id: Optional[str], kind: str) -> dict:
        """Return a reputation record for an entity.

        Always returns a dict. `known` is False and `first_time` is True when the
        entity has no prior history — the caller decides how to price that.
        """
        if not entity_id:
            return {"entity_id": entity_id, "kind": kind, "known": False,
                    "first_time": True, "risk": self.UNKNOWN_RISK,
                    "fraud_reports": 0, "times_seen": 0, "note": "no id supplied"}
        rec = self.entities.get(entity_id)
        if rec is None:
            return {"entity_id": entity_id, "kind": kind, "known": False,
                    "first_time": True, "risk": self.UNKNOWN_RISK,
                    "fraud_reports": 0, "times_seen": 0, "note": "first sighting"}
        return {**rec, "known": True, "first_time": rec["times_seen"] == 0}

    def observe(self, entity_id: Optional[str], kind: str):
        """Record a benign sighting so an entity builds a track record."""
        if not entity_id:
            return
        rec = self.entities.get(entity_id)
        now = time.time()
        if rec is None:
            self.entities[entity_id] = {
                "entity_id": entity_id, "kind": kind, "risk": self.UNKNOWN_RISK,
                "fraud_reports": 0, "times_seen": 1,
                "first_seen": now, "last_updated": now, "note": "observed",
                "source": "runtime",
            }
        else:
            rec["times_seen"] += 1
            rec["last_updated"] = now
        self._persist(self.entities[entity_id])

    def report_fraud(self, entity_id: Optional[str], kind: str, severity: float = 30.0) -> Optional[dict]:
        """Elevate an entity's reputation after a confirmed-fraud / blocked txn.

        This is the write side of the blockchain feedback loop (wired in Phase 3):
        the fraud fingerprint is committed so future transfers to the same
        entity inherit the elevated risk.
        """
        if not entity_id:
            return None
        now = time.time()
        rec = self.entities.get(entity_id)
        if rec is None:
            rec = {"entity_id": entity_id, "kind": kind, "risk": 0.0,
                   "fraud_reports": 0, "times_seen": 0,
                   "first_seen": now, "note": "reported", "source": "runtime"}
            self.entities[entity_id] = rec
        rec["risk"] = float(min(100.0, rec["risk"] + severity))
        rec["fraud_reports"] += 1
        rec["times_seen"] += 1
        rec["last_updated"] = now
        self._persist(rec)
        return rec

    def summary(self) -> dict:
        beneficiaries = [e for e in self.entities.values() if e["kind"] == "beneficiary"]
        devices = [e for e in self.entities.values() if e["kind"] == "device"]
        flagged = [e for e in self.entities.values() if e["risk"] >= 50]
        return {
            "total_entities": len(self.entities),
            "beneficiaries": len(beneficiaries),
            "devices": len(devices),
            "flagged": len(flagged),
            "entries": sorted(self.entities.values(), key=lambda e: e["risk"], reverse=True)[:25],
        }
