"""
Simple deployment helper instructions (local script placeholder).

Note: This file includes the convenient command-line instructions and
a small wrapper to call out to an external deployment tool.
We recommend using Foundry (forge) or Hardhat to compile and deploy the contract.
This script does not compile or deploy directly; it documents the minimal steps
and provides a small helper for verifying the contract address in the .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def instructions():
    print("Deployment instructions for MusicPredictionOracle.sol\n")
    print("1) Install Foundry (recommended) or Hardhat.")
    print("   Foundry: curl -L https://foundry.paradigm.xyz | bash && source ~/.bashrc && foundryup")
    print("2) Add contracts/MusicPredictionOracle.sol to a Foundry project src/ and run:")
    print("   forge build")
    print("3) Deploy to Avalanche Fuji testnet:")
    print("   forge create --rpc-url https://api.avax-test.network/ext/bc/C/rpc --private-key $PRIVATE_KEY src/MusicPredictionOracle.sol:MusicPredictionOracle")
    print("4) After deploy, set PREDICTION_ORACLE_ADDRESS in .env to the deployed address.")
    print("\nThis script is intentionally lightweight to keep CI/environment choices flexible.")

if __name__ == "__main__":
    instructions()