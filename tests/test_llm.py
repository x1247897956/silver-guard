"""HTTP failure classification uses real client parsing with a local transport."""

import httpx
import pytest

from silverguard.llm import LLMClient, LLMError


@pytest.mark.parametrize('status', [400, 401, 402, 403, 404])
def test_permanent_http_error_is_not_retried(status, monkeypatch):
    requests = []
    monkeypatch.setattr('silverguard.llm.time.sleep', lambda _: None)

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={'error': 'unavailable'})

    with LLMClient(api_key='test', model='test', base_url='https://example.invalid',
                   transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(LLMError, match=f'HTTP {status}'):
            client.complete(system='test', user='test')
        assert len(requests) == 1
        assert client.failures == 1


def test_transient_http_error_retries_then_recovers(monkeypatch):
    requests = []
    monkeypatch.setattr('silverguard.llm.time.sleep', lambda _: None)

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={'error': 'rate limited'})
        return httpx.Response(200, json={
            'model': 'reported-test', 'choices': [{'message': {'content': '{}'}}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
        })

    with LLMClient(api_key='test', model='test', base_url='https://example.invalid',
                   transport=httpx.MockTransport(respond)) as client:
        response = client.complete(system='test', user='test')
        assert len(requests) == 2
        assert response.attempts == 2
        assert client.failures == 0
