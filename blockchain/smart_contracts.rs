// Smart Contract Engine for Medical Data Blockchain
// Supports automated data access control and audit logging

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContract {
    pub contract_id: String,
    pub contract_type: ContractType,
    pub owner: String,
    pub created_at: u64,
    pub state: ContractState,
    pub code_hash: String,
    pub storage: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContractType {
    DataAccessControl,
    ConsentManagement,
    AuditLog,
    DataSharing,
    ResearchAgreement,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContractState {
    Active,
    Paused,
    Terminated,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAccessRequest {
    pub request_id: String,
    pub requester: String,
    pub patient_id: String,
    pub data_type: String,
    pub purpose: String,
    pub timestamp: u64,
    pub approved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub consent_id: String,
    pub patient_id: String,
    pub granted_to: Vec<String>,
    pub data_categories: Vec<String>,
    pub expiry_date: u64,
    pub revocable: bool,
    pub active: bool,
}

pub struct SmartContractEngine {
    contracts: HashMap<String, SmartContract>,
    access_requests: Vec<DataAccessRequest>,
    consent_records: HashMap<String, ConsentRecord>,
}

impl SmartContractEngine {
    pub fn new() -> Self {
        SmartContractEngine {
            contracts: HashMap::new(),
            access_requests: Vec::new(),
            consent_records: HashMap::new(),
        }
    }

    pub fn deploy_contract(
        &mut self,
        contract_type: ContractType,
        owner: String,
        code: &str,
    ) -> Result<String, String> {
        let contract_id = self.generate_contract_id(&owner, &contract_type);
        let code_hash = self.hash_code(code);
        
        let contract = SmartContract {
            contract_id: contract_id.clone(),
            contract_type,
            owner,
            created_at: Self::current_timestamp(),
            state: ContractState::Active,
            code_hash,
            storage: HashMap::new(),
        };
        
        self.contracts.insert(contract_id.clone(), contract);
        
        println!("[SmartContract] Deployed contract: {}", contract_id);
        Ok(contract_id)
    }

    pub fn execute_data_access_request(
        &mut self,
        requester: String,
        patient_id: String,
        data_type: String,
        purpose: String,
    ) -> Result<bool, String> {
        // Check if consent exists
        let has_consent = self.check_consent(&patient_id, &requester, &data_type)?;
        
        if !has_consent {
            return Err("No valid consent found".to_string());
        }
        
        // Create access request
        let request = DataAccessRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            requester: requester.clone(),
            patient_id: patient_id.clone(),
            data_type: data_type.clone(),
            purpose: purpose.clone(),
            timestamp: Self::current_timestamp(),
            approved: true,
        };
        
        self.access_requests.push(request);
        
        // Log to audit trail
        self.log_audit_event(
            &format!("Data access granted to {} for patient {}", requester, patient_id),
        );
        
        Ok(true)
    }

    pub fn grant_consent(
        &mut self,
        patient_id: String,
        granted_to: Vec<String>,
        data_categories: Vec<String>,
        duration_days: u64,
    ) -> Result<String, String> {
        let consent_id = uuid::Uuid::new_v4().to_string();
        let expiry_date = Self::current_timestamp() + (duration_days * 86400);
        
        let consent = ConsentRecord {
            consent_id: consent_id.clone(),
            patient_id: patient_id.clone(),
            granted_to,
            data_categories,
            expiry_date,
            revocable: true,
            active: true,
        };
        
        self.consent_records.insert(consent_id.clone(), consent);
        
        self.log_audit_event(
            &format!("Consent granted by patient {}", patient_id),
        );
        
        Ok(consent_id)
    }

    pub fn revoke_consent(&mut self, consent_id: &str) -> Result<(), String> {
        if let Some(consent) = self.consent_records.get_mut(consent_id) {
            if !consent.revocable {
                return Err("Consent is not revocable".to_string());
            }
            
            consent.active = false;
            
            self.log_audit_event(
                &format!("Consent {} revoked by patient {}", consent_id, consent.patient_id),
            );
            
            Ok(())
        } else {
            Err("Consent not found".to_string())
        }
    }

    fn check_consent(
        &self,
        patient_id: &str,
        requester: &str,
        data_type: &str,
    ) -> Result<bool, String> {
        let current_time = Self::current_timestamp();
        
        for consent in self.consent_records.values() {
            if consent.patient_id == patient_id
                && consent.active
                && consent.expiry_date > current_time
                && consent.granted_to.contains(&requester.to_string())
                && consent.data_categories.contains(&data_type.to_string())
            {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    pub fn get_audit_trail(&self, patient_id: &str) -> Vec<DataAccessRequest> {
        self.access_requests
            .iter()
            .filter(|req| req.patient_id == patient_id)
            .cloned()
            .collect()
    }

    pub fn execute_research_agreement(
        &mut self,
        researcher_id: String,
        data_requirements: Vec<String>,
        duration_months: u64,
    ) -> Result<String, String> {
        let contract_id = self.deploy_contract(
            ContractType::ResearchAgreement,
            researcher_id.clone(),
            "research_agreement_v1",
        )?;
        
        if let Some(contract) = self.contracts.get_mut(&contract_id) {
            contract.storage.insert("researcher".to_string(), researcher_id);
            contract.storage.insert(
                "data_requirements".to_string(),
                data_requirements.join(","),
            );
            contract.storage.insert(
                "expiry".to_string(),
                (Self::current_timestamp() + duration_months * 30 * 86400).to_string(),
            );
        }
        
        Ok(contract_id)
    }

    pub fn verify_contract_integrity(&self, contract_id: &str, code: &str) -> bool {
        if let Some(contract) = self.contracts.get(contract_id) {
            let code_hash = self.hash_code(code);
            return code_hash == contract.code_hash;
        }
        false
    }

    fn log_audit_event(&self, event: &str) {
        println!("[Audit] {} - {}", Self::current_timestamp(), event);
    }

    fn generate_contract_id(&self, owner: &str, contract_type: &ContractType) -> String {
        let input = format!("{:?}_{}_{}" , contract_type, owner, Self::current_timestamp());
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    fn hash_code(&self, code: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(code.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    pub fn get_active_contracts(&self) -> Vec<&SmartContract> {
        self.contracts
            .values()
            .filter(|c| c.state == ContractState::Active)
            .collect()
    }

    pub fn pause_contract(&mut self, contract_id: &str) -> Result<(), String> {
        if let Some(contract) = self.contracts.get_mut(contract_id) {
            contract.state = ContractState::Paused;
            Ok(())
        } else {
            Err("Contract not found".to_string())
        }
    }

    pub fn terminate_contract(&mut self, contract_id: &str) -> Result<(), String> {
        if let Some(contract) = self.contracts.get_mut(contract_id) {
            contract.state = ContractState::Terminated;
            self.log_audit_event(&format!("Contract {} terminated", contract_id));
            Ok(())
        } else {
            Err("Contract not found".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deploy_contract() {
        let mut engine = SmartContractEngine::new();
        let result = engine.deploy_contract(
            ContractType::DataAccessControl,
            "owner1".to_string(),
            "contract_code",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_consent_management() {
        let mut engine = SmartContractEngine::new();
        let consent_id = engine.grant_consent(
            "patient123".to_string(),
            vec!["doctor1".to_string()],
            vec!["medical_records".to_string()],
            30,
        ).unwrap();
        
        assert!(engine.consent_records.contains_key(&consent_id));
    }

    #[test]
    fn test_data_access_with_consent() {
        let mut engine = SmartContractEngine::new();
        
        // Grant consent first
        engine.grant_consent(
            "patient123".to_string(),
            vec!["researcher1".to_string()],
            vec!["imaging_data".to_string()],
            30,
        ).unwrap();
        
        // Request access
        let result = engine.execute_data_access_request(
            "researcher1".to_string(),
            "patient123".to_string(),
            "imaging_data".to_string(),
            "Clinical research".to_string(),
        );
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
    }

    #[test]
    fn test_revoke_consent() {
        let mut engine = SmartContractEngine::new();
        
        let consent_id = engine.grant_consent(
            "patient123".to_string(),
            vec!["doctor1".to_string()],
            vec!["medical_records".to_string()],
            30,
        ).unwrap();
        
        let result = engine.revoke_consent(&consent_id);
        assert!(result.is_ok());
        
        let consent = engine.consent_records.get(&consent_id).unwrap();
        assert_eq!(consent.active, false);
    }
}
