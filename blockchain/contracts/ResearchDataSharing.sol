// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ResearchDataSharing
 * @dev Smart contract for managing anonymized research data sharing
 * @notice Implements k-anonymity and differential privacy controls
 */
contract ResearchDataSharing {
    
    // Events
    event DatasetPublished(bytes32 indexed datasetId, address indexed publisher, uint256 recordCount);
    event DatasetAccessed(bytes32 indexed datasetId, address indexed researcher, uint256 timestamp);
    event DUACreated(bytes32 indexed duaId, address indexed researcher, bytes32 indexed datasetId);
    event PrivacyViolationDetected(bytes32 indexed datasetId, string violationType);
    
    // Structs
    struct Dataset {
        bytes32 datasetId;
        address publisher;
        string name;
        string description;
        uint256 recordCount;
        uint256 kAnonymity;
        uint256 epsilon; // Differential privacy parameter (scaled by 1000)
        bytes32 dataHash;
        uint256 publishedAt;
        bool isActive;
        string[] allowedPurposes;
    }
    
    struct DataUseAgreement {
        bytes32 duaId;
        address researcher;
        bytes32 datasetId;
        string purpose;
        uint256 startDate;
        uint256 endDate;
        bool isActive;
        string[] restrictions;
    }
    
    struct ResearcherProfile {
        address researcherAddress;
        string name;
        string institution;
        string[] certifications;
        uint256 reputation;
        bool isVerified;
        uint256 registeredAt;
    }
    
    struct AccessLog {
        bytes32 logId;
        address researcher;
        bytes32 datasetId;
        uint256 timestamp;
        string queryType;
        uint256 recordsAccessed;
    }
    
    // State variables
    mapping(bytes32 => Dataset) public datasets;
    mapping(bytes32 => DataUseAgreement) public duas;
    mapping(address => ResearcherProfile) public researchers;
    mapping(bytes32 => AccessLog[]) public accessLogs;
    mapping(bytes32 => mapping(address => bool)) public datasetAccess;
    
    bytes32[] public datasetIds;
    address public admin;
    uint256 public minKAnonymity = 5;
    uint256 public maxEpsilon = 1000; // 1.0 scaled by 1000
    
    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    
    modifier onlyVerifiedResearcher() {
        require(researchers[msg.sender].isVerified, "Researcher not verified");
        _;
    }
    
    modifier hasValidDUA(bytes32 _datasetId) {
        bytes32 duaId = keccak256(abi.encodePacked(msg.sender, _datasetId));
        DataUseAgreement memory dua = duas[duaId];
        require(dua.isActive, "No active DUA");
        require(block.timestamp >= dua.startDate && block.timestamp <= dua.endDate, "DUA expired");
        _;
    }
    
    constructor() {
        admin = msg.sender;
    }
    
    /**
     * @dev Register as researcher
     */
    function registerResearcher(
        string memory _name,
        string memory _institution,
        string[] memory _certifications
    ) public {
        require(researchers[msg.sender].researcherAddress == address(0), "Already registered");
        
        researchers[msg.sender] = ResearcherProfile({
            researcherAddress: msg.sender,
            name: _name,
            institution: _institution,
            certifications: _certifications,
            reputation: 100,
            isVerified: false,
            registeredAt: block.timestamp
        });
    }
    
    /**
     * @dev Verify researcher (admin only)
     */
    function verifyResearcher(address _researcher) public onlyAdmin {
        require(researchers[_researcher].researcherAddress != address(0), "Researcher not found");
        researchers[_researcher].isVerified = true;
    }
    
    /**
     * @dev Publish anonymized dataset
     */
    function publishDataset(
        string memory _name,
        string memory _description,
        uint256 _recordCount,
        uint256 _kAnonymity,
        uint256 _epsilon,
        bytes32 _dataHash,
        string[] memory _allowedPurposes
    ) public onlyVerifiedResearcher {
        require(_kAnonymity >= minKAnonymity, "K-anonymity too low");
        require(_epsilon <= maxEpsilon, "Epsilon too high");
        
        bytes32 datasetId = keccak256(abi.encodePacked(
            msg.sender,
            _name,
            block.timestamp
        ));
        
        datasets[datasetId] = Dataset({
            datasetId: datasetId,
            publisher: msg.sender,
            name: _name,
            description: _description,
            recordCount: _recordCount,
            kAnonymity: _kAnonymity,
            epsilon: _epsilon,
            dataHash: _dataHash,
            publishedAt: block.timestamp,
            isActive: true,
            allowedPurposes: _allowedPurposes
        });
        
        datasetIds.push(datasetId);
        
        emit DatasetPublished(datasetId, msg.sender, _recordCount);
    }
    
    /**
     * @dev Create Data Use Agreement
     */
    function createDUA(
        bytes32 _datasetId,
        string memory _purpose,
        uint256 _duration,
        string[] memory _restrictions
    ) public onlyVerifiedResearcher {
        require(datasets[_datasetId].isActive, "Dataset not active");
        
        // Check if purpose is allowed
        bool purposeAllowed = false;
        for (uint i = 0; i < datasets[_datasetId].allowedPurposes.length; i++) {
            if (keccak256(bytes(datasets[_datasetId].allowedPurposes[i])) == keccak256(bytes(_purpose))) {
                purposeAllowed = true;
                break;
            }
        }
        require(purposeAllowed, "Purpose not allowed");
        
        bytes32 duaId = keccak256(abi.encodePacked(msg.sender, _datasetId));
        
        duas[duaId] = DataUseAgreement({
            duaId: duaId,
            researcher: msg.sender,
            datasetId: _datasetId,
            purpose: _purpose,
            startDate: block.timestamp,
            endDate: block.timestamp + _duration,
            isActive: true,
            restrictions: _restrictions
        });
        
        datasetAccess[_datasetId][msg.sender] = true;
        
        emit DUACreated(duaId, msg.sender, _datasetId);
    }
    
    /**
     * @dev Access dataset (logs access)
     */
    function accessDataset(
        bytes32 _datasetId,
        string memory _queryType,
        uint256 _recordsAccessed
    ) public hasValidDUA(_datasetId) {
        bytes32 logId = keccak256(abi.encodePacked(
            msg.sender,
            _datasetId,
            block.timestamp
        ));
        
        AccessLog memory log = AccessLog({
            logId: logId,
            researcher: msg.sender,
            datasetId: _datasetId,
            timestamp: block.timestamp,
            queryType: _queryType,
            recordsAccessed: _recordsAccessed
        });
        
        accessLogs[_datasetId].push(log);
        
        emit DatasetAccessed(_datasetId, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Revoke DUA
     */
    function revokeDUA(bytes32 _duaId) public {
        DataUseAgreement storage dua = duas[_duaId];
        require(
            msg.sender == dua.researcher || 
            msg.sender == datasets[dua.datasetId].publisher || 
            msg.sender == admin,
            "Not authorized"
        );
        
        dua.isActive = false;
        datasetAccess[dua.datasetId][dua.researcher] = false;
    }
    
    /**
     * @dev Get dataset details
     */
    function getDataset(bytes32 _datasetId) public view returns (Dataset memory) {
        return datasets[_datasetId];
    }
    
    /**
     * @dev Get all datasets
     */
    function getAllDatasets() public view returns (bytes32[] memory) {
        return datasetIds;
    }
    
    /**
     * @dev Get access logs for dataset
     */
    function getAccessLogs(bytes32 _datasetId) public view returns (AccessLog[] memory) {
        return accessLogs[_datasetId];
    }
    
    /**
     * @dev Update researcher reputation
     */
    function updateReputation(address _researcher, uint256 _newReputation) public onlyAdmin {
        require(researchers[_researcher].researcherAddress != address(0), "Researcher not found");
        researchers[_researcher].reputation = _newReputation;
    }
    
    /**
     * @dev Deactivate dataset
     */
    function deactivateDataset(bytes32 _datasetId) public {
        require(
            msg.sender == datasets[_datasetId].publisher || msg.sender == admin,
            "Not authorized"
        );
        datasets[_datasetId].isActive = false;
    }
}
