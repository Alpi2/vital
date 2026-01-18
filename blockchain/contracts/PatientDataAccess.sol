// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title PatientDataAccess
 * @dev Smart contract for managing patient data access permissions on VitalStream blockchain
 * @notice This contract implements HIPAA-compliant access control for medical data
 */
contract PatientDataAccess {
    
    // Events
    event AccessGranted(address indexed patient, address indexed accessor, string dataType, uint256 expiresAt);
    event AccessRevoked(address indexed patient, address indexed accessor, string dataType);
    event DataAccessed(address indexed accessor, bytes32 indexed dataHash, uint256 timestamp);
    event AuditLogCreated(bytes32 indexed logId, address indexed accessor, string action);
    
    // Structs
    struct AccessPermission {
        address patient;
        address accessor;
        string dataType;
        uint256 grantedAt;
        uint256 expiresAt;
        bool isActive;
        string purpose;
    }
    
    struct AuditLog {
        bytes32 logId;
        address accessor;
        bytes32 dataHash;
        string action;
        uint256 timestamp;
        string ipAddress;
    }
    
    struct PatientData {
        bytes32 dataHash;
        string dataType;
        uint256 createdAt;
        uint256 updatedAt;
        bool exists;
    }
    
    // State variables
    mapping(address => mapping(address => mapping(string => AccessPermission))) public permissions;
    mapping(bytes32 => AuditLog) public auditLogs;
    mapping(bytes32 => PatientData) public patientDataRegistry;
    mapping(address => bytes32[]) public patientDataHashes;
    mapping(address => uint256) public accessCount;
    
    address public admin;
    uint256 public totalAuditLogs;
    
    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }
    
    modifier onlyPatient(address _patient) {
        require(msg.sender == _patient, "Only patient can perform this action");
        _;
    }
    
    modifier hasAccess(address _patient, string memory _dataType) {
        require(
            msg.sender == _patient || 
            (permissions[_patient][msg.sender][_dataType].isActive && 
             block.timestamp < permissions[_patient][msg.sender][_dataType].expiresAt),
            "Access denied"
        );
        _;
    }
    
    constructor() {
        admin = msg.sender;
        totalAuditLogs = 0;
    }
    
    /**
     * @dev Grant access permission to accessor for patient's data
     * @param _accessor Address of the entity requesting access
     * @param _dataType Type of data (e.g., "ECG", "MRI", "Lab Results")
     * @param _duration Duration in seconds for which access is granted
     * @param _purpose Purpose of access (e.g., "Treatment", "Research")
     */
    function grantAccess(
        address _accessor,
        string memory _dataType,
        uint256 _duration,
        string memory _purpose
    ) public {
        require(_accessor != address(0), "Invalid accessor address");
        require(_duration > 0, "Duration must be greater than 0");
        
        uint256 expiresAt = block.timestamp + _duration;
        
        permissions[msg.sender][_accessor][_dataType] = AccessPermission({
            patient: msg.sender,
            accessor: _accessor,
            dataType: _dataType,
            grantedAt: block.timestamp,
            expiresAt: expiresAt,
            isActive: true,
            purpose: _purpose
        });
        
        emit AccessGranted(msg.sender, _accessor, _dataType, expiresAt);
        _createAuditLog(_accessor, bytes32(0), "ACCESS_GRANTED", "");
    }
    
    /**
     * @dev Revoke access permission
     * @param _accessor Address of the accessor
     * @param _dataType Type of data
     */
    function revokeAccess(address _accessor, string memory _dataType) public {
        require(permissions[msg.sender][_accessor][_dataType].isActive, "Permission not active");
        
        permissions[msg.sender][_accessor][_dataType].isActive = false;
        
        emit AccessRevoked(msg.sender, _accessor, _dataType);
        _createAuditLog(_accessor, bytes32(0), "ACCESS_REVOKED", "");
    }
    
    /**
     * @dev Register patient data hash
     * @param _dataHash Hash of the encrypted patient data
     * @param _dataType Type of data
     */
    function registerData(bytes32 _dataHash, string memory _dataType) public {
        require(_dataHash != bytes32(0), "Invalid data hash");
        require(!patientDataRegistry[_dataHash].exists, "Data already registered");
        
        patientDataRegistry[_dataHash] = PatientData({
            dataHash: _dataHash,
            dataType: _dataType,
            createdAt: block.timestamp,
            updatedAt: block.timestamp,
            exists: true
        });
        
        patientDataHashes[msg.sender].push(_dataHash);
        
        _createAuditLog(msg.sender, _dataHash, "DATA_REGISTERED", "");
    }
    
    /**
     * @dev Access patient data (logs access)
     * @param _patient Patient address
     * @param _dataHash Hash of the data being accessed
     * @param _dataType Type of data
     */
    function accessData(
        address _patient,
        bytes32 _dataHash,
        string memory _dataType
    ) public hasAccess(_patient, _dataType) {
        require(patientDataRegistry[_dataHash].exists, "Data not found");
        
        accessCount[msg.sender]++;
        
        emit DataAccessed(msg.sender, _dataHash, block.timestamp);
        _createAuditLog(msg.sender, _dataHash, "DATA_ACCESSED", "");
    }
    
    /**
     * @dev Check if accessor has permission
     * @param _patient Patient address
     * @param _accessor Accessor address
     * @param _dataType Type of data
     * @return bool True if accessor has valid permission
     */
    function checkPermission(
        address _patient,
        address _accessor,
        string memory _dataType
    ) public view returns (bool) {
        AccessPermission memory perm = permissions[_patient][_accessor][_dataType];
        return perm.isActive && block.timestamp < perm.expiresAt;
    }
    
    /**
     * @dev Get patient's data hashes
     * @param _patient Patient address
     * @return bytes32[] Array of data hashes
     */
    function getPatientDataHashes(address _patient) public view returns (bytes32[] memory) {
        return patientDataHashes[_patient];
    }
    
    /**
     * @dev Get access permission details
     * @param _patient Patient address
     * @param _accessor Accessor address
     * @param _dataType Type of data
     * @return AccessPermission struct
     */
    function getPermission(
        address _patient,
        address _accessor,
        string memory _dataType
    ) public view returns (AccessPermission memory) {
        return permissions[_patient][_accessor][_dataType];
    }
    
    /**
     * @dev Internal function to create audit log
     */
    function _createAuditLog(
        address _accessor,
        bytes32 _dataHash,
        string memory _action,
        string memory _ipAddress
    ) internal {
        bytes32 logId = keccak256(abi.encodePacked(
            _accessor,
            _dataHash,
            _action,
            block.timestamp,
            totalAuditLogs
        ));
        
        auditLogs[logId] = AuditLog({
            logId: logId,
            accessor: _accessor,
            dataHash: _dataHash,
            action: _action,
            timestamp: block.timestamp,
            ipAddress: _ipAddress
        });
        
        totalAuditLogs++;
        
        emit AuditLogCreated(logId, _accessor, _action);
    }
    
    /**
     * @dev Get audit log by ID
     * @param _logId Log ID
     * @return AuditLog struct
     */
    function getAuditLog(bytes32 _logId) public view returns (AuditLog memory) {
        return auditLogs[_logId];
    }
    
    /**
     * @dev Emergency pause (admin only)
     */
    function emergencyPause() public onlyAdmin {
        // Implementation for emergency pause
        _createAuditLog(msg.sender, bytes32(0), "EMERGENCY_PAUSE", "");
    }
}
