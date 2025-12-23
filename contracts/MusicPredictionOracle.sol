// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title MusicPredictionOracle
 * @dev Smart contract for storing and verifying music hit predictions on Avalanche
 * @notice This contract allows storing ML predictions about song success and later verification
 */
contract MusicPredictionOracle {
    
    struct Prediction {
        address predictor;          // Address that made the prediction
        uint256 timestamp;          // When the prediction was made
        uint256 songId;            // Unique identifier for the song
        uint256 hitProbability;    // Hit probability (0-10000 representing 0-100.00%)
        bool isPredictedHit;       // True if predicted as hit (>50%)
        string modelVersion;       // Version of the ML model used
        string songHash;           // Hash of song features for verification
        bool verified;             // Whether the prediction has been verified
        bool actualOutcome;        // Actual outcome (set during verification)
    }
    
    // State variables
    mapping(uint256 => Prediction) public predictions;
    mapping(address => uint256[]) public predictorHistory;
    uint256 public nextPredictionId;
    address public owner;
    
    // Events
    event PredictionStored(
        uint256 indexed predictionId,
        address indexed predictor,
        uint256 songId,
        uint256 hitProbability,
        bool isPredictedHit,
        string modelVersion
    );
    
    event PredictionVerified(
        uint256 indexed predictionId,
        bool actualOutcome,
        address indexed verifier
    );
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier validPrediction(uint256 _predictionId) {
        require(_predictionId < nextPredictionId, "Prediction does not exist");
        _;
    }
    
    /**
     * @dev Constructor sets the deployer as owner
     */
    constructor() {
        owner = msg.sender;
        nextPredictionId = 0;
    }
    
    /**
     * @dev Store a new music hit prediction
     * @param _songId Unique identifier for the song
     * @param _hitProbability Hit probability (0-10000 representing 0-100.00%)
     * @param _isPredictedHit True if predicted as hit
     * @param _modelVersion Version of the ML model used
     * @param _songHash Hash of song features for verification
     * @return predictionId The ID of the stored prediction
     */
    function storePrediction(
        uint256 _songId,
        uint256 _hitProbability,
        bool _isPredictedHit,
        string memory _modelVersion,
        string memory _songHash
    ) external returns (uint256) {
        require(_hitProbability <= 10000, "Hit probability must be <= 10000");
        require(bytes(_modelVersion).length > 0, "Model version cannot be empty");
        require(bytes(_songHash).length > 0, "Song hash cannot be empty");
        
        uint256 predictionId = nextPredictionId;
        
        predictions[predictionId] = Prediction({
            predictor: msg.sender,
            timestamp: block.timestamp,
            songId: _songId,
            hitProbability: _hitProbability,
            isPredictedHit: _isPredictedHit,
            modelVersion: _modelVersion,
            songHash: _songHash,
            verified: false,
            actualOutcome: false
        });
        
        predictorHistory[msg.sender].push(predictionId);
        nextPredictionId++;
        
        emit PredictionStored(
            predictionId,
            msg.sender,
            _songId,
            _hitProbability,
            _isPredictedHit,
            _modelVersion
        );
        
        return predictionId;
    }
    
    /**
     * @dev Verify a prediction with actual outcome
     * @param _predictionId ID of the prediction to verify
     * @param _actualOutcome The actual outcome (true if song was a hit)
     */
    function verifyPrediction(uint256 _predictionId, bool _actualOutcome) 
        external 
        onlyOwner 
        validPrediction(_predictionId) 
    {
        require(!predictions[_predictionId].verified, "Prediction already verified");
        
        predictions[_predictionId].verified = true;
        predictions[_predictionId].actualOutcome = _actualOutcome;
        
        emit PredictionVerified(_predictionId, _actualOutcome, msg.sender);
    }
    
    /**
     * @dev Get prediction details
     * @param _predictionId ID of the prediction
     * @return All prediction details
     */
    function getPrediction(uint256 _predictionId) 
        external 
        view 
        validPrediction(_predictionId) 
        returns (Prediction memory) 
    {
        return predictions[_predictionId];
    }
    
    /**
     * @dev Get prediction history for a specific predictor
     * @param _predictor Address of the predictor
     * @return Array of prediction IDs
     */
    function getPredictorHistory(address _predictor) 
        external 
        view 
        returns (uint256[] memory) 
    {
        return predictorHistory[_predictor];
    }
    
    /**
     * @dev Calculate accuracy for a predictor
     * @param _predictor Address of the predictor
     * @return accuracy Percentage accuracy (0-10000 representing 0-100.00%)
     * @return totalVerified Number of verified predictions
     */
    function calculatePredictorAccuracy(address _predictor) 
        external 
        view 
        returns (uint256 accuracy, uint256 totalVerified) 
    {
        uint256[] memory history = predictorHistory[_predictor];
        uint256 correct = 0;
        totalVerified = 0;
        
        for (uint256 i = 0; i < history.length; i++) {
            Prediction memory pred = predictions[history[i]];
            if (pred.verified) {
                totalVerified++;
                if (pred.isPredictedHit == pred.actualOutcome) {
                    correct++;
                }
            }
        }
        
        if (totalVerified == 0) {
            return (0, 0);
        }
        
        accuracy = (correct * 10000) / totalVerified;
        return (accuracy, totalVerified);
    }
    
    /**
     * @dev Transfer ownership to a new owner
     * @param _newOwner Address of the new owner
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "New owner cannot be zero address");
        owner = _newOwner;
    }
    
    /**
     * @dev Get total number of predictions stored
     * @return Total number of predictions
     */
    function getTotalPredictions() external view returns (uint256) {
        return nextPredictionId;
    }
}