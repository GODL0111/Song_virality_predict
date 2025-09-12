# Avalanche Integration – Song Virality Predictor

This folder adds a lightweight Avalanche integration to the Song Virality Predictor project.
It provides:
- A Solidity smart contract (MusicPredictionOracle.sol) to store and later verify predictions on-chain.
- A Python integration (avalanche_song_predictor.py) that extends your existing SongHitPredictor to publish predictions to Avalanche.
- A small deployment helper and configuration files.

What I added (proposed)
- contracts/MusicPredictionOracle.sol — smart contract for storing & verifying predictions
- avalanche_config.py — Avalanche + wallet config (pulls secrets from .env)
- avalanche_song_predictor.py — Python class that calls your model and stores predictions on-chain
- deploy_contract.py — deployment instructions helper
- requirements_blockchain.txt — extra pip deps
- .env.example — example env file
- README_AVALANCHE_INTEGRATION.md — this doc

Quick start (dev / testnet)
1. Install dependencies
   pip install -r requirements_blockchain.txt

2. Create .env from .env.example and add your private key and wallet address. Use the Fuji testnet for initial testing.

3. Compile and deploy the smart contract (Foundry or Hardhat recommended)
   - Example with Foundry:
     - Install Foundry: curl -L https://foundry.paradigm.xyz | bash && source ~/.bashrc && foundryup
     - Put MusicPredictionOracle.sol in a Foundry project `src/`
     - Run: forge build
     - Deploy:
       forge create --rpc-url https://api.avax-test.network/ext/bc/C/rpc \
         --private-key $AVALANCHE_PRIVATE_KEY \
         src/MusicPredictionOracle.sol:MusicPredictionOracle

4. Update .env PREDICTION_ORACLE_ADDRESS with deployed address.

5. Run a demo:
   - Ensure your model is trained and model files exist (use your existing train flow).
   - Run a small script or import AvalancheSongPredictor and call `predict_and_store_on_avalanche()`.

Security & Compliance notes
- Never store private keys in source files or commit .env to git.
- Use testnet (Fuji) for development; only deploy to mainnet after security audits.
- This flow is focused on analytics / verification — avoid gambling mechanics.

Next steps (what I'll push if you confirm)
- Add the files above into feature/avalanche-integration branch.
- Create a small demo script that runs a prediction and stores it on-chain using a demo song.
- Add a short PR description and open a pull request against main.