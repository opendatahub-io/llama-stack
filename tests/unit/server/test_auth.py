# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llama_stack.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    CustomAuthConfig,
    OAuth2IntrospectionConfig,
    OAuth2JWKSConfig,
    OAuth2TokenAuthConfig,
)
from llama_stack.core.request_headers import User
from llama_stack.core.server.auth import AuthenticationMiddleware, _has_required_scope
from llama_stack.core.server.auth_providers import (
    get_attributes_from_claims,
)


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP error: {self.status_code}")


@pytest.fixture
def mock_auth_endpoint():
    return "http://mock-auth-service/validate"


@pytest.fixture
def valid_api_key():
    return "valid_api_key_12345"


@pytest.fixture
def invalid_api_key():
    return "invalid_api_key_67890"


@pytest.fixture
def valid_token():
    return "valid.jwt.token"


@pytest.fixture
def invalid_token():
    return "invalid.jwt.token"


@pytest.fixture
def http_app(mock_auth_endpoint):
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=CustomAuthConfig(
            type=AuthProviderType.CUSTOM,
            endpoint=mock_auth_endpoint,
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def http_client(http_app):
    return TestClient(http_app)


@pytest.fixture
def mock_scope():
    return {
        "type": "http",
        "path": "/models/list",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test.jwt.token"),
            (b"user-agent", b"test-user-agent"),
        ],
        "query_string": b"limit=100&offset=0",
    }


@pytest.fixture
def mock_http_middleware(mock_auth_endpoint):
    mock_app = AsyncMock()
    auth_config = AuthenticationConfig(
        provider_config=CustomAuthConfig(
            type=AuthProviderType.CUSTOM,
            endpoint=mock_auth_endpoint,
        ),
        access_policy=[],
    )
    return AuthenticationMiddleware(mock_app, auth_config, {}), mock_app


@pytest.fixture
def mock_impls():
    """Mock implementations for scope testing"""
    return {}


@pytest.fixture
def middleware_with_mocks(mock_auth_endpoint):
    """Create AuthenticationMiddleware with mocked route implementations"""
    mock_app = AsyncMock()
    auth_config = AuthenticationConfig(
        provider_config=CustomAuthConfig(
            type=AuthProviderType.CUSTOM,
            endpoint=mock_auth_endpoint,
        ),
        access_policy=[],
    )
    middleware = AuthenticationMiddleware(mock_app, auth_config, {})

    # Mock the route_impls to simulate finding routes with required scopes
    from llama_stack.schema_utils import WebMethod

    routes = {
        ("POST", "/test/scoped"): WebMethod(route="/test/scoped", method="POST", required_scope="test.read"),
        ("GET", "/test/public"): WebMethod(route="/test/public", method="GET"),
        ("GET", "/health"): WebMethod(route="/health", method="GET", require_authentication=False),
        ("GET", "/version"): WebMethod(route="/version", method="GET", require_authentication=False),
        ("GET", "/models/list"): WebMethod(route="/models/list", method="GET", require_authentication=True),
    }

    # Mock the route finding logic
    def mock_find_matching_route(method, path, route_impls):
        webmethod = routes.get((method, path))
        if webmethod:
            return None, {}, path, webmethod
        raise ValueError("No matching route")

    import llama_stack.core.server.auth

    llama_stack.core.server.auth.find_matching_route = mock_find_matching_route
    llama_stack.core.server.auth.initialize_route_impls = lambda impls: {}

    return middleware, mock_app


async def mock_post_success(*args, **kwargs):
    return MockResponse(
        200,
        {
            "message": "Authentication successful",
            "principal": "test-principal",
            "attributes": {
                "roles": ["admin", "user"],
                "teams": ["ml-team", "nlp-team"],
                "projects": ["llama-3", "project-x"],
                "namespaces": ["research", "production"],
            },
        },
    )


async def mock_post_failure(*args, **kwargs):
    return MockResponse(401, {"message": "Authentication failed"})


async def mock_post_exception(*args, **kwargs):
    raise Exception("Connection error")


async def mock_post_success_with_scope(*args, **kwargs):
    """Mock auth response for user with test.read scope"""
    return MockResponse(
        200,
        {
            "message": "Authentication successful",
            "principal": "test-user",
            "attributes": {
                "scopes": ["test.read", "other.scope"],
                "roles": ["user"],
            },
        },
    )


async def mock_post_success_no_scope(*args, **kwargs):
    """Mock auth response for user without required scope"""
    return MockResponse(
        200,
        {
            "message": "Authentication successful",
            "principal": "test-user",
            "attributes": {
                "scopes": ["other.scope"],
                "roles": ["user"],
            },
        },
    )


# HTTP Endpoint Tests
def test_missing_auth_header(http_client):
    response = http_client.get("/test")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["error"]["message"]
    assert "validated by mock-auth-service" in response.json()["error"]["message"]


def test_invalid_auth_header_format(http_client):
    response = http_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Invalid Authorization header format" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_post_success)
def test_valid_http_authentication(http_client, valid_api_key):
    response = http_client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.post", new=mock_post_failure)
def test_invalid_http_authentication(http_client, invalid_api_key):
    response = http_client.get("/test", headers={"Authorization": f"Bearer {invalid_api_key}"})
    assert response.status_code == 401
    assert "Authentication failed" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_post_exception)
def test_http_auth_service_error(http_client, valid_api_key):
    response = http_client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 401
    assert "Authentication service error" in response.json()["error"]["message"]


def test_http_auth_request_payload(http_client, valid_api_key, mock_auth_endpoint):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MockResponse(200, {"message": "Authentication successful"})
        mock_post.return_value = mock_response

        http_client.get(
            "/test?param1=value1&param2=value2",
            headers={
                "Authorization": f"Bearer {valid_api_key}",
                "User-Agent": "TestClient",
                "Content-Type": "application/json",
            },
        )

        # Check that the auth endpoint was called with the correct payload
        call_args = mock_post.call_args
        assert call_args is not None

        url, kwargs = call_args[0][0], call_args[1]
        assert url == mock_auth_endpoint

        payload = kwargs["json"]
        assert payload["api_key"] == valid_api_key
        assert payload["request"]["path"] == "/test"
        assert "authorization" not in payload["request"]["headers"]
        assert "param1" in payload["request"]["params"]
        assert "param2" in payload["request"]["params"]


async def test_http_middleware_with_access_attributes(mock_http_middleware, mock_scope):
    """Test HTTP middleware behavior with access attributes"""
    middleware, mock_app = mock_http_middleware
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MockResponse(
            200,
            {
                "message": "Authentication successful",
                "principal": "test-principal",
                "attributes": {
                    "roles": ["admin", "user"],
                    "teams": ["ml-team", "nlp-team"],
                    "projects": ["llama-3", "project-x"],
                    "namespaces": ["research", "production"],
                },
            },
        )
        mock_post.return_value = mock_response

        await middleware(mock_scope, mock_receive, mock_send)

        assert "user_attributes" in mock_scope
        attributes = mock_scope["user_attributes"]
        assert attributes["roles"] == ["admin", "user"]
        assert attributes["teams"] == ["ml-team", "nlp-team"]
        assert attributes["projects"] == ["llama-3", "project-x"]
        assert attributes["namespaces"] == ["research", "production"]

        mock_app.assert_called_once_with(mock_scope, mock_receive, mock_send)


# oauth2 token provider tests


@pytest.fixture
def oauth2_app():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            jwks=OAuth2JWKSConfig(
                uri="http://mock-authz-service/token/introspect",
            ),
            audience="llama-stack",
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def oauth2_client(oauth2_app):
    return TestClient(oauth2_app)


def test_missing_auth_header_oauth2(oauth2_client):
    response = oauth2_client.get("/test")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["error"]["message"]
    assert "OAuth2 Bearer token" in response.json()["error"]["message"]


def test_invalid_auth_header_format_oauth2(oauth2_client):
    response = oauth2_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Invalid Authorization header format" in response.json()["error"]["message"]


async def mock_jwks_response(*args, **kwargs):
    return MockResponse(
        200,
        {
            "keys": [
                {
                    "kid": "1234567890",
                    "kty": "oct",
                    "alg": "HS256",
                    "use": "sig",
                    "k": base64.b64encode(b"foobarbaz").decode(),
                }
            ]
        },
    )


@pytest.fixture
def jwt_token_valid():
    import jwt

    return jwt.encode(
        {
            "sub": "my-user",
            "groups": ["group1", "group2"],
            "scope": "foo bar",
            "aud": "llama-stack",
        },
        key="foobarbaz",
        algorithm="HS256",
        headers={"kid": "1234567890"},
    )


@pytest.fixture
def mock_jwks_urlopen():
    """Mock urllib.request.urlopen for PyJWKClient JWKS requests."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        # Mock the JWKS response for PyJWKClient
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "keys": [
                    {
                        "kid": "1234567890",
                        "kty": "oct",
                        "alg": "HS256",
                        "use": "sig",
                        "k": base64.b64encode(b"foobarbaz").decode(),
                    }
                ]
            }
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        yield mock_urlopen


def test_valid_oauth2_authentication(oauth2_client, jwt_token_valid, mock_jwks_urlopen):
    response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.get", new=mock_jwks_response)
def test_invalid_oauth2_authentication(oauth2_client, invalid_token):
    response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {invalid_token}"})
    assert response.status_code == 401
    assert "Invalid JWT token" in response.json()["error"]["message"]


async def mock_auth_jwks_response(*args, **kwargs):
    if "headers" not in kwargs or "Authorization" not in kwargs["headers"]:
        return MockResponse(401, {})
    authz = kwargs["headers"]["Authorization"]
    if authz != "Bearer my-jwks-token":
        return MockResponse(401, {})
    return await mock_jwks_response(args, kwargs)


@pytest.fixture
def oauth2_app_with_jwks_token():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            jwks=OAuth2JWKSConfig(
                uri="http://mock-authz-service/token/introspect",
                key_recheck_period=3600,
                token="my-jwks-token",
            ),
            audience="llama-stack",
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def oauth2_client_with_jwks_token(oauth2_app_with_jwks_token):
    return TestClient(oauth2_app_with_jwks_token)


@patch("httpx.AsyncClient.get", new=mock_auth_jwks_response)
def test_oauth2_with_jwks_token_expected(oauth2_client, jwt_token_valid):
    response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
    assert response.status_code == 401


def test_oauth2_with_jwks_token_configured(oauth2_client_with_jwks_token, jwt_token_valid, mock_jwks_urlopen):
    response = oauth2_client_with_jwks_token.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


def test_get_attributes_from_claims():
    claims = {
        "sub": "my-user",
        "groups": ["group1", "group2"],
        "scope": "foo bar",
        "aud": "llama-stack",
    }
    attributes = get_attributes_from_claims(claims, {"sub": "roles", "groups": "teams"})
    assert attributes["roles"] == ["my-user"]
    assert attributes["teams"] == ["group1", "group2"]

    claims = {
        "sub": "my-user",
        "tenant": "my-tenant",
    }
    attributes = get_attributes_from_claims(claims, {"sub": "roles", "tenant": "namespaces"})
    assert attributes["roles"] == ["my-user"]
    assert attributes["namespaces"] == ["my-tenant"]

    claims = {
        "sub": "my-user",
        "username": "my-username",
        "tenant": "my-tenant",
        "groups": ["group1", "group2"],
        "team": "my-team",
    }
    attributes = get_attributes_from_claims(
        claims,
        {
            "sub": "roles",
            "tenant": "namespaces",
            "username": "roles",
            "team": "teams",
            "groups": "teams",
        },
    )
    assert set(attributes["roles"]) == {"my-user", "my-username"}
    assert set(attributes["teams"]) == {"my-team", "group1", "group2"}
    assert attributes["namespaces"] == ["my-tenant"]


# TODO: add more tests for oauth2 token provider


# oauth token introspection tests
@pytest.fixture
def mock_introspection_endpoint():
    return "http://mock-authz-service/token/introspect"


@pytest.fixture
def introspection_app(mock_introspection_endpoint):
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            introspection=OAuth2IntrospectionConfig(
                url=mock_introspection_endpoint,
                client_id="myclient",
                client_secret="abcdefg",
            ),
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def introspection_app_with_custom_mapping(mock_introspection_endpoint):
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            introspection=OAuth2IntrospectionConfig(
                url=mock_introspection_endpoint,
                client_id="myclient",
                client_secret="abcdefg",
                send_secret_in_body=True,
            ),
            claims_mapping={
                "sub": "roles",
                "scope": "roles",
                "groups": "teams",
                "aud": "namespaces",
            },
        ),
        access_policy=[],
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def introspection_client(introspection_app):
    return TestClient(introspection_app)


@pytest.fixture
def introspection_client_with_custom_mapping(introspection_app_with_custom_mapping):
    return TestClient(introspection_app_with_custom_mapping)


def test_missing_auth_header_introspection(introspection_client):
    response = introspection_client.get("/test")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["error"]["message"]
    assert "OAuth2 Bearer token" in response.json()["error"]["message"]


def test_invalid_auth_header_format_introspection(introspection_client):
    response = introspection_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Invalid Authorization header format" in response.json()["error"]["message"]


async def mock_introspection_active(*args, **kwargs):
    return MockResponse(
        200,
        {
            "active": True,
            "sub": "my-user",
            "groups": ["group1", "group2"],
            "scope": "foo bar",
            "aud": ["set1", "set2"],
        },
    )


async def mock_introspection_inactive(*args, **kwargs):
    return MockResponse(
        200,
        {
            "active": False,
        },
    )


async def mock_introspection_invalid(*args, **kwargs):
    class InvalidResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            raise ValueError("Not JSON")

    return InvalidResponse(200)


async def mock_introspection_failed(*args, **kwargs):
    return MockResponse(
        500,
        {},
    )


@patch("httpx.AsyncClient.post", new=mock_introspection_active)
def test_valid_introspection_authentication(introspection_client, valid_api_key):
    response = introspection_client.get("/test", headers={"Authorization": f"Bearer {valid_api_key}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.post", new=mock_introspection_inactive)
def test_inactive_introspection_authentication(introspection_client, invalid_api_key):
    response = introspection_client.get("/test", headers={"Authorization": f"Bearer {invalid_api_key}"})
    assert response.status_code == 401
    assert "Token not active" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_introspection_invalid)
def test_invalid_introspection_authentication(introspection_client, invalid_api_key):
    response = introspection_client.get("/test", headers={"Authorization": f"Bearer {invalid_api_key}"})
    assert response.status_code == 401
    assert "Not JSON" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_introspection_failed)
def test_failed_introspection_authentication(introspection_client, invalid_api_key):
    response = introspection_client.get("/test", headers={"Authorization": f"Bearer {invalid_api_key}"})
    assert response.status_code == 401
    assert "Token introspection failed: 500" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_introspection_active)
def test_valid_introspection_with_custom_mapping_authentication(
    introspection_client_with_custom_mapping, valid_api_key
):
    response = introspection_client_with_custom_mapping.get(
        "/test", headers={"Authorization": f"Bearer {valid_api_key}"}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


# Scope-based authorization tests
@patch("httpx.AsyncClient.post", new=mock_post_success_with_scope)
async def test_scope_authorization_success(middleware_with_mocks, valid_api_key):
    """Test that user with required scope can access protected endpoint"""
    middleware, mock_app = middleware_with_mocks
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    scope = {
        "type": "http",
        "path": "/test/scoped",
        "method": "POST",
        "headers": [(b"authorization", f"Bearer {valid_api_key}".encode())],
    }

    await middleware(scope, mock_receive, mock_send)

    # Should call the downstream app (no 403 error sent)
    mock_app.assert_called_once_with(scope, mock_receive, mock_send)
    mock_send.assert_not_called()


@patch("httpx.AsyncClient.post", new=mock_post_success_no_scope)
async def test_scope_authorization_denied(middleware_with_mocks, valid_api_key):
    """Test that user without required scope gets 403 access denied"""
    middleware, mock_app = middleware_with_mocks
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    scope = {
        "type": "http",
        "path": "/test/scoped",
        "method": "POST",
        "headers": [(b"authorization", f"Bearer {valid_api_key}".encode())],
    }

    await middleware(scope, mock_receive, mock_send)

    # Should send 403 error, not call downstream app
    mock_app.assert_not_called()
    assert mock_send.call_count == 2  # start + body

    # Check the response
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["status"] == 403

    body_call = mock_send.call_args_list[1][0][0]
    body_text = body_call["body"].decode()
    assert "Access denied" in body_text
    assert "test.read" in body_text


@patch("httpx.AsyncClient.post", new=mock_post_success_no_scope)
async def test_public_endpoint_no_scope_required(middleware_with_mocks, valid_api_key):
    """Test that public endpoints work without specific scopes"""
    middleware, mock_app = middleware_with_mocks
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    scope = {
        "type": "http",
        "path": "/test/public",
        "method": "GET",
        "headers": [(b"authorization", f"Bearer {valid_api_key}".encode())],
    }

    await middleware(scope, mock_receive, mock_send)

    # Should call the downstream app (no error)
    mock_app.assert_called_once_with(scope, mock_receive, mock_send)
    mock_send.assert_not_called()


async def test_scope_authorization_no_auth_disabled(middleware_with_mocks):
    """Test that when auth is disabled (no user), scope checks are bypassed"""
    middleware, mock_app = middleware_with_mocks
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    scope = {
        "type": "http",
        "path": "/test/scoped",
        "method": "POST",
        "headers": [],  # No authorization header
    }

    await middleware(scope, mock_receive, mock_send)

    # Should send 401 auth error, not call downstream app
    mock_app.assert_not_called()
    assert mock_send.call_count == 2  # start + body

    # Check the response
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["status"] == 401

    body_call = mock_send.call_args_list[1][0][0]
    body_text = body_call["body"].decode()
    assert "Authentication required" in body_text


def test_has_required_scope_function():
    """Test the _has_required_scope function directly"""
    # Test user with required scope
    user_with_scope = User(principal="test-user", attributes={"scopes": ["test.read", "other.scope"]})
    assert _has_required_scope("test.read", user_with_scope)

    # Test user without required scope
    user_without_scope = User(principal="test-user", attributes={"scopes": ["other.scope"]})
    assert not _has_required_scope("test.read", user_without_scope)

    # Test user with no scopes attribute
    user_no_scopes = User(principal="test-user", attributes={})
    assert not _has_required_scope("test.read", user_no_scopes)

    # Test no user (auth disabled)
    assert _has_required_scope("test.read", None)


@pytest.fixture
def mock_kubernetes_api_server():
    return "https://api.cluster.example.com:6443"


@pytest.fixture
def kubernetes_auth_app(mock_kubernetes_api_server):
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config={
            "type": "kubernetes",
            "api_server_url": mock_kubernetes_api_server,
            "verify_tls": False,
            "claims_mapping": {
                "username": "roles",
                "groups": "roles",
                "uid": "uid_attr",
            },
        },
    )
    app.add_middleware(AuthenticationMiddleware, auth_config=auth_config, impls={})

    @app.get("/test")
    def test_endpoint():
        return {"message": "Authentication successful"}

    return app


@pytest.fixture
def kubernetes_auth_client(kubernetes_auth_app):
    return TestClient(kubernetes_auth_app)


def test_missing_auth_header_kubernetes_auth(kubernetes_auth_client):
    response = kubernetes_auth_client.get("/test")
    assert response.status_code == 401
    assert "Authentication required" in response.json()["error"]["message"]


def test_invalid_auth_header_format_kubernetes_auth(kubernetes_auth_client):
    response = kubernetes_auth_client.get("/test", headers={"Authorization": "InvalidFormat token123"})
    assert response.status_code == 401
    assert "Invalid Authorization header format" in response.json()["error"]["message"]


async def mock_kubernetes_selfsubjectreview_success(*args, **kwargs):
    return MockResponse(
        201,
        {
            "apiVersion": "authentication.k8s.io/v1",
            "kind": "SelfSubjectReview",
            "metadata": {"creationTimestamp": "2025-07-15T13:53:56Z"},
            "status": {
                "userInfo": {
                    "username": "alice",
                    "uid": "alice-uid-123",
                    "groups": ["system:authenticated", "developers", "admins"],
                    "extra": {"scopes.authorization.openshift.io": ["user:full"]},
                }
            },
        },
    )


async def mock_kubernetes_selfsubjectreview_failure(*args, **kwargs):
    return MockResponse(401, {"message": "Unauthorized"})


async def mock_kubernetes_selfsubjectreview_http_error(*args, **kwargs):
    return MockResponse(500, {"message": "Internal Server Error"})


@patch("httpx.AsyncClient.post", new=mock_kubernetes_selfsubjectreview_success)
def test_valid_kubernetes_auth_authentication(kubernetes_auth_client, valid_token):
    response = kubernetes_auth_client.get("/test", headers={"Authorization": f"Bearer {valid_token}"})
    assert response.status_code == 200
    assert response.json() == {"message": "Authentication successful"}


@patch("httpx.AsyncClient.post", new=mock_kubernetes_selfsubjectreview_failure)
def test_invalid_kubernetes_auth_authentication(kubernetes_auth_client, invalid_token):
    response = kubernetes_auth_client.get("/test", headers={"Authorization": f"Bearer {invalid_token}"})
    assert response.status_code == 401
    assert "Invalid token" in response.json()["error"]["message"]


@patch("httpx.AsyncClient.post", new=mock_kubernetes_selfsubjectreview_http_error)
def test_kubernetes_auth_http_error(kubernetes_auth_client, valid_token):
    response = kubernetes_auth_client.get("/test", headers={"Authorization": f"Bearer {valid_token}"})
    assert response.status_code == 401
    assert "Token validation failed" in response.json()["error"]["message"]


def test_kubernetes_auth_request_payload(kubernetes_auth_client, valid_token, mock_kubernetes_api_server):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MockResponse(
            200,
            {
                "apiVersion": "authentication.k8s.io/v1",
                "kind": "SelfSubjectReview",
                "metadata": {"creationTimestamp": "2025-07-15T13:53:56Z"},
                "status": {
                    "userInfo": {
                        "username": "test-user",
                        "uid": "test-uid",
                        "groups": ["test-group"],
                    }
                },
            },
        )
        mock_post.return_value = mock_response

        kubernetes_auth_client.get("/test", headers={"Authorization": f"Bearer {valid_token}"})

        # Verify the request was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL (passed as positional argument)
        assert call_args[0][0] == f"{mock_kubernetes_api_server}/apis/authentication.k8s.io/v1/selfsubjectreviews"

        # Check headers (passed as keyword argument)
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {valid_token}"
        assert headers["Content-Type"] == "application/json"

        # Check request body (passed as keyword argument)
        request_body = call_args[1]["json"]
        assert request_body["apiVersion"] == "authentication.k8s.io/v1"
        assert request_body["kind"] == "SelfSubjectReview"


async def test_unauthenticated_endpoint_access_health(middleware_with_mocks):
    """Test that /health endpoints can be accessed without authentication"""
    middleware, mock_app = middleware_with_mocks

    # Test request to /health without auth header (level prefix v1 is added by router)
    scope = {"type": "http", "path": "/health", "headers": [], "method": "GET"}
    receive = AsyncMock()
    send = AsyncMock()

    # Should allow the request to proceed without authentication
    await middleware(scope, receive, send)

    # Verify that the request was passed to the app
    mock_app.assert_called_once_with(scope, receive, send)

    # Verify that no error response was sent
    assert not any(call[0][0].get("status") == 401 for call in send.call_args_list)


async def test_unauthenticated_endpoint_denied_for_other_paths(middleware_with_mocks):
    """Test that endpoints other than /health and /version require authentication"""
    middleware, mock_app = middleware_with_mocks

    # Test request to /models/list without auth header
    scope = {"type": "http", "path": "/models/list", "headers": [], "method": "GET"}
    receive = AsyncMock()
    send = AsyncMock()

    # Should return 401 error
    await middleware(scope, receive, send)

    # Verify that the app was NOT called
    mock_app.assert_not_called()

    # Verify that a 401 error response was sent
    assert any(call[0][0].get("status") == 401 for call in send.call_args_list)
