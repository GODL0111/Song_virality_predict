// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title Music Prediction Oracle
/// @notice Stores ML predictions (immutable after storing) and allows verification later.
///         Probabilities and confidences are stored as basis points (0 - 10000) representing 0.00% - 100.00%.
contract MusicPredictionOracle {
    struct Prediction {
        string songId;
        uint256 hitProbability;  // basis points: 0 - 10000 (0.00% - 100.00%)
        uint256 confidence;      // basis points
        string modelVersion;
        uint256 timestamp;
        address predictor;
        bool verified;
        uint256 actualResult;    // basis points, set when verification occurs
    }

    mapping(string => Prediction) private predictions;
    mapping(address => bool) public authorizedPredictors;
    address public owner;

    event PredictionStored(
        string indexed songId,
        uint256 hitProbability,
        uint256 confidence,
        address indexed predictor,
        uint256 timestamp
    );

    event PredictionVerified(
        string indexed songId,
        uint256 actualResult,
        bool wasAccurate
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyAuthorized() {
        require(authorizedPredictors[msg.sender], "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedPredictors[msg.sender] = true;
    }

    function setAuthorized(address account, bool allowed) external onlyOwner {
        authorizedPredictors[account] = allowed;
    }

    /// @notice Store a prediction on-chain. Overwrites previous entry for same songId.
    function storePrediction(
        string calldata songId,
        uint256 hitProbability,
        uint256 confidence,
        string calldata modelVersion
    ) external onlyAuthorized {
        require(bytes(songId).length > 0, "Invalid songId");
        require(hitProbability <= 10000, "Probability out of range");
        require(confidence <= 10000, "Confidence out of range");

        predictions[songId] = Prediction({
            songId: songId,
            hitProbability: hitProbability,
            confidence: confidence,
            modelVersion: modelVersion,
            timestamp: block.timestamp,
            predictor: msg.sender,
            verified: false,
            actualResult: 0
        });

        emit PredictionStored(songId, hitProbability, confidence, msg.sender, block.timestamp);
    }

    /// @notice Read prediction for a song
    function getPrediction(string calldata songId)
        external
        view
        returns (
            string memory,
            uint256,
            uint256,
            string memory,
            uint256,
            address,
            bool,
            uint256
        )
    {
        Prediction storage p = predictions[songId];
        return (
            p.songId,
            p.hitProbability,
            p.confidence,
            p.modelVersion,
            p.timestamp,
            p.predictor,
            p.verified,
            p.actualResult
        );
    }

    /// @notice Verify prediction after real-world performance known.
    /// @dev Only authorized callers (owner or designated verifier) should call this.
    function verifyPrediction(string calldata songId, uint256 actualResult) external onlyAuthorized {
        Prediction storage p = predictions[songId];
        require(p.timestamp != 0, "Prediction not found");
        require(!p.verified, "Already verified");
        require(actualResult <= 10000, "actualResult out of range");

        p.actualResult = actualResult;
        p.verified = true;

        // simple accuracy test: within 15% relative tolerance
        uint256 predicted = p.hitProbability;
        if (predicted == 0) {
            emit PredictionVerified(songId, actualResult, actualResult == 0);
            return;
        }
        uint256 diff = predicted > actualResult ? predicted - actualResult : actualResult - predicted;
        // relative difference in basis points = diff * 10000 / predicted
        uint256 relBp = (diff * 10000) / predicted;
        bool accurate = relBp <= 1500; // 15% tolerance

        emit PredictionVerified(songId, actualResult, accurate);
    }
}
