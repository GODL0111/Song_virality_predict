import json
import time
import hashlib
from datetime import datetime
from web3 import Web3
from predict_V0.2 import SongHitPredictor
from avalanche_config import AVALANCHE_CONFIG, WALLET_CONFIG, CONTRACT_ADDRESSES

class AvalancheSongPredictor(SongHitPredictor):
    def __init__(self, network="TESTNET", model_dir="models", data_dir="data"):
        super().__init__(model_dir, data_dir)
        self.network = network
        self.config = AVALANCHE_CONFIG[network]
        self.w3 = Web3(Web3.HTTPProvider(self.config["RPC_URL"]))

        # wallet
        self.private_key = WALLET_CONFIG["PRIVATE_KEY"]
        if not self.private_key:
            raise EnvironmentError("AVALANCHE_PRIVATE_KEY not set in .env")
        self.account = self.w3.eth.account.from_key(self.private_key)
        self.wallet_address = self.account.address

        # contract (ABI minimal for storePrediction and getPrediction)
        self.oracle_abi = [
            {
                "inputs": [
                    {"internalType": "string", "name": "songId", "type": "string"},
                    {"internalType": "uint256", "name": "hitProbability", "type": "uint256"},
                    {"internalType": "uint256", "name": "confidence", "type": "uint256"},
                    {"internalType": "string", "name": "modelVersion", "type": "string"}
                ],
                "name": "storePrediction",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "string", "name": "songId", "type": "string"}],
                "name": "getPrediction",
                "outputs": [
                    {"internalType": "string", "name": "", "type": "string"},
                    {"internalType": "uint256", "name": "", "type": "uint256"},
                    {"internalType": "uint256", "name": "", "type": "uint256"},
                    {"internalType": "string", "name": "", "type": "string"},
                    {"internalType": "uint256", "name": "", "type": "uint256"},
                    {"internalType": "address", "name": "", "type": "address"},
                    {"internalType": "bool", "name": "", "type": "bool"},
                    {"internalType": "uint256", "name": "", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]

        oracle_addr = CONTRACT_ADDRESSES.get("PREDICTION_ORACLE")
        if oracle_addr and oracle_addr != "0x...":
            self.oracle_contract = self.w3.eth.contract(address=oracle_addr, abi=self.oracle_abi)
        else:
            self.oracle_contract = None

    def _generate_song_id(self, song_features):
        feature_string = json.dumps(song_features, sort_keys=True)
        song_hash = hashlib.md5(feature_string.encode()).hexdigest()
        return f"song_{song_hash[:12]}"

    def predict_and_store_on_avalanche(self, song_features, song_id=None):
        prediction_result = self.predict_song_hit_probability(song_features)
        if not prediction_result:
            return None

        if not song_id:
            song_id = self._generate_song_id(song_features)

        hit_probability_bp = int(prediction_result['hit_probability'] * 10000)
        confidence_bp = int(prediction_result['confidence'] * 10000)
        model_version = self.model_metadata.get('created_at', 'v1.0')

        if not self.oracle_contract:
            raise RuntimeError("Oracle contract not configured. Set PREDICTION_ORACLE_ADDRESS in .env")

        txn = self.oracle_contract.functions.storePrediction(
            song_id, hit_probability_bp, confidence_bp, model_version
        ).build_transaction({
            "from": self.wallet_address,
            "nonce": self.w3.eth.get_transaction_count(self.wallet_address),
            "gas": 200000,
            "gasPrice": self.w3.to_wei("25", "gwei")
        })

        signed = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "song_id": song_id,
            "tx_hash": tx_hash.hex(),
            "receipt": dict(receipt),
            "ml_prediction": prediction_result,
            "explorer_url": f"{self.config['EXPLORER']}/tx/{tx_hash.hex()}"
        }

    def get_prediction_from_chain(self, song_id):
        if not self.oracle_contract:
            raise RuntimeError("Oracle contract not configured.")
        res = self.oracle_contract.functions.getPrediction(song_id).call()
        return {
            "song_id": res[0],
            "hit_probability": res[1] / 10000,
            "confidence": res[2] / 10000,
            "model_version": res[3],
            "timestamp": res[4],
            "predictor": res[5],
            "verified": res[6],
            "actual_result": (res[7] / 10000) if res[7] > 0 else None
        }