import os
from dotenv import load_dotenv

load_dotenv()

# Avalanche Network Configuration
AVALANCHE_CONFIG = {
    # Fuji Testnet (development)
    "TESTNET": {
        "RPC_URL": "https://api.avax-test.network/ext/bc/C/rpc",
        "CHAIN_ID": 43113,
        "EXPLORER": "https://testnet.snowtrace.io",
        "NAME": "Avalanche Fuji Testnet"
    },
    # Mainnet (production)
    "MAINNET": {
        "RPC_URL": "https://api.avax.network/ext/bc/C/rpc",
        "CHAIN_ID": 43114,
        "EXPLORER": "https://snowtrace.io",
        "NAME": "Avalanche Mainnet"
    }
}

# Wallet config (pull from .env)
WALLET_CONFIG = {
    "PRIVATE_KEY": os.getenv("AVALANCHE_PRIVATE_KEY"),
    "ADDRESS": os.getenv("WALLET_ADDRESS")
}

# Placeholder contract addresses - update after deployment
CONTRACT_ADDRESSES = {
    "PREDICTION_ORACLE": os.getenv("PREDICTION_ORACLE_ADDRESS", "0x..."),
}