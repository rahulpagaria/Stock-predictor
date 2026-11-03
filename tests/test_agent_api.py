import json
import unittest
from unittest.mock import patch

from stock_predictor.agent_api import AgentApiError, HttpAgentApiAdapter


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class AgentApiTests(unittest.TestCase):
    @patch("stock_predictor.agent_api.urlopen")
    def test_posts_json_and_returns_json_object(self, urlopen: object) -> None:
        urlopen.return_value = FakeResponse(b'{"accepted": true}')
        adapter = HttpAgentApiAdapter("https://agent.example/predict")
        payload = {"ticker": "AAPL", "signal": "FLAT"}

        response = adapter.send_prediction(payload)

        self.assertEqual(response, {"accepted": True})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data.decode("utf-8")), payload)

    @patch("stock_predictor.agent_api.urlopen")
    def test_rejects_non_object_response(self, urlopen: object) -> None:
        urlopen.return_value = FakeResponse(b"[]")
        adapter = HttpAgentApiAdapter("https://agent.example/predict")

        with self.assertRaisesRegex(AgentApiError, "JSON object"):
            adapter.send_prediction({"ticker": "AAPL"})


if __name__ == "__main__":
    unittest.main()
