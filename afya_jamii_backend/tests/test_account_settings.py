"""Tests for the account settings endpoints.

Account deletion gets the most attention here. It is irreversible and it
destroys health records, so the tests assert both halves: that it refuses
without correct confirmation, and that when it does run it removes the child
rows rather than orphaning them.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.database import engine
from app.main import app
from app.models import ConversationHistory, UserDB, VitalsRecord

PASSWORD = "Mimba2026!"
VITALS = {
    "age": 29,
    "systolic_bp": 124,
    "diastolic_bp": 80,
    "bs": 5.4,
    "body_temp": 36.9,
    "body_temp_unit": "celsius",
    "heart_rate": 76,
    "patient_history": "Routine check",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def account(client):
    """A fresh signed-in account, unique per test."""
    username = f"test_{uuid.uuid4().hex[:10]}"
    payload = {
        "username": username,
        "email": f"{username}@example.co.ke",
        "full_name": "Test User",
        "account_type": "pregnant",
        "password": PASSWORD,
    }
    assert client.post("/api/v1/auth/signup", json=payload).status_code == 201

    login = client.post(
        "/api/v1/auth/login", json={"username": username, "password": PASSWORD}
    )
    assert login.status_code == 200

    token = login.json()["access_token"]
    return {
        "username": username,
        "email": payload["email"],
        "headers": {"Authorization": f"Bearer {token}"},
    }


# ── Profile ────────────────────────────────────────────────────────────────

def test_profile_is_readable(client, account):
    response = client.get("/api/v1/users/me", headers=account["headers"])
    assert response.status_code == 200
    assert response.json()["username"] == account["username"]


def test_profile_update_applies_only_given_fields(client, account):
    response = client.patch(
        "/api/v1/users/me",
        json={"full_name": "Amina Wanjiku"},
        headers=account["headers"],
    )
    assert response.status_code == 200

    body = response.json()
    assert body["full_name"] == "Amina Wanjiku"
    # Untouched fields must survive a partial update.
    assert body["account_type"] == "pregnant"
    assert body["email"] == account["email"]


def test_account_type_can_be_changed(client, account):
    response = client.patch(
        "/api/v1/users/me", json={"account_type": "postnatal"}, headers=account["headers"]
    )
    assert response.status_code == 200
    assert response.json()["account_type"] == "postnatal"


def test_empty_update_is_rejected(client, account):
    assert client.patch("/api/v1/users/me", json={}, headers=account["headers"]).status_code == 422


def test_email_cannot_collide_with_another_account(client, account):
    other = f"test_{uuid.uuid4().hex[:10]}"
    client.post(
        "/api/v1/auth/signup",
        json={
            "username": other,
            "email": f"{other}@example.co.ke",
            "full_name": "Other",
            "account_type": "general",
            "password": PASSWORD,
        },
    )

    response = client.patch(
        "/api/v1/users/me",
        json={"email": f"{other}@example.co.ke"},
        headers=account["headers"],
    )
    assert response.status_code == 409


def test_profile_requires_authentication(client):
    assert client.get("/api/v1/users/me").status_code == 401
    assert client.patch("/api/v1/users/me", json={"full_name": "x"}).status_code == 401


# ── Password ───────────────────────────────────────────────────────────────

def test_password_change_requires_the_current_password(client, account):
    response = client.post(
        "/api/v1/users/me/password",
        json={"current_password": "wrong-password", "new_password": "Newpass123!"},
        headers=account["headers"],
    )
    assert response.status_code == 403


def test_password_change_then_sign_in_with_the_new_one(client, account):
    new_password = "Kabisa2026#"

    response = client.post(
        "/api/v1/users/me/password",
        json={"current_password": PASSWORD, "new_password": new_password},
        headers=account["headers"],
    )
    assert response.status_code == 204

    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": account["username"], "password": new_password},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": account["username"], "password": PASSWORD},
        ).status_code
        == 401
    )


def test_weak_new_password_is_rejected(client, account):
    for weak in ("short", "12345678", "onlyletters"):
        response = client.post(
            "/api/v1/users/me/password",
            json={"current_password": PASSWORD, "new_password": weak},
            headers=account["headers"],
        )
        assert response.status_code == 422, weak


def test_new_password_must_differ(client, account):
    response = client.post(
        "/api/v1/users/me/password",
        json={"current_password": PASSWORD, "new_password": PASSWORD},
        headers=account["headers"],
    )
    assert response.status_code == 422


# ── Deactivation ───────────────────────────────────────────────────────────

def test_deactivation_blocks_sign_in_but_keeps_records(client, account):
    client.post("/api/v1/vitals/submit", json={"vitals": VITALS}, headers=account["headers"])

    assert (
        client.post("/api/v1/users/me/deactivate", headers=account["headers"]).status_code == 204
    )

    # Sign-in is refused, and the existing token no longer works either.
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": account["username"], "password": PASSWORD},
        ).status_code
        == 401
    )
    assert client.get("/api/v1/users/me", headers=account["headers"]).status_code == 403

    # The health records are still there — that is the point of deactivating
    # rather than deleting.
    with Session(engine) as session:
        user = session.exec(
            select(UserDB).where(UserDB.username == account["username"])
        ).first()
        assert user is not None
        assert user.is_active is False
        assert len(session.exec(select(VitalsRecord).where(VitalsRecord.user_id == user.id)).all()) == 1


# ── Deletion ───────────────────────────────────────────────────────────────

def test_deletion_requires_the_correct_password(client, account):
    response = client.request(
        "DELETE",
        "/api/v1/users/me",
        json={"password": "not-the-password", "confirmation": "DELETE MY ACCOUNT"},
        headers=account["headers"],
    )
    assert response.status_code == 403
    assert client.get("/api/v1/users/me", headers=account["headers"]).status_code == 200


@pytest.mark.parametrize(
    "phrase",
    ["", "delete my account", "DELETE", "DELETE MY ACCOUNT PLEASE", "yes"],
)
def test_deletion_requires_the_exact_confirmation_phrase(client, account, phrase):
    response = client.request(
        "DELETE",
        "/api/v1/users/me",
        json={"password": PASSWORD, "confirmation": phrase},
        headers=account["headers"],
    )
    assert response.status_code == 422
    assert client.get("/api/v1/users/me", headers=account["headers"]).status_code == 200


def test_deletion_removes_the_user_and_every_child_row(client, account):
    """The whole point: no orphaned health records are left behind."""
    client.post("/api/v1/vitals/submit", json={"vitals": VITALS}, headers=account["headers"])
    client.post("/api/v1/vitals/submit", json={"vitals": VITALS}, headers=account["headers"])

    with Session(engine) as session:
        user = session.exec(
            select(UserDB).where(UserDB.username == account["username"])
        ).first()
        user_id = user.id
        assert len(session.exec(select(VitalsRecord).where(VitalsRecord.user_id == user_id)).all()) == 2

    response = client.request(
        "DELETE",
        "/api/v1/users/me",
        json={"password": PASSWORD, "confirmation": "DELETE MY ACCOUNT"},
        headers=account["headers"],
    )
    assert response.status_code == 200

    body = response.json()
    assert body["username"] == account["username"]
    assert body["vitals_records_deleted"] == 2

    with Session(engine) as session:
        assert session.exec(select(UserDB).where(UserDB.id == user_id)).first() is None
        assert session.exec(select(VitalsRecord).where(VitalsRecord.user_id == user_id)).all() == []
        assert (
            session.exec(
                select(ConversationHistory).where(ConversationHistory.user_id == user_id)
            ).all()
            == []
        )


def test_deleted_account_cannot_sign_in_again(client, account):
    client.request(
        "DELETE",
        "/api/v1/users/me",
        json={"password": PASSWORD, "confirmation": "DELETE MY ACCOUNT"},
        headers=account["headers"],
    )

    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": account["username"], "password": PASSWORD},
        ).status_code
        == 401
    )
    # The token is now signed for a user that no longer exists.
    assert client.get("/api/v1/users/me", headers=account["headers"]).status_code == 401


def test_deletion_frees_the_username_and_email(client, account):
    client.request(
        "DELETE",
        "/api/v1/users/me",
        json={"password": PASSWORD, "confirmation": "DELETE MY ACCOUNT"},
        headers=account["headers"],
    )

    # A deleted account must not leave its unique constraints occupied.
    response = client.post(
        "/api/v1/auth/signup",
        json={
            "username": account["username"],
            "email": account["email"],
            "full_name": "Someone Else",
            "account_type": "general",
            "password": PASSWORD,
        },
    )
    assert response.status_code == 201


def test_deletion_requires_authentication(client):
    response = client.request(
        "DELETE",
        "/api/v1/users/me",
        json={"password": PASSWORD, "confirmation": "DELETE MY ACCOUNT"},
    )
    assert response.status_code == 401


# ── Username changes ───────────────────────────────────────────────────────

def test_username_can_be_changed_and_a_new_token_is_issued(client, account):
    new_name = f"renamed_{uuid.uuid4().hex[:8]}"

    response = client.patch(
        "/api/v1/users/me", json={"username": new_name}, headers=account["headers"]
    )
    assert response.status_code == 200

    body = response.json()
    assert body["username"] == new_name
    # The old token names the previous username as its subject, so a
    # replacement has to come back or the caller is locked out mid-edit.
    assert body["access_token"], "no replacement token was issued"

    fresh = {"Authorization": f"Bearer {body['access_token']}"}
    assert client.get("/api/v1/users/me", headers=fresh).json()["username"] == new_name

    # And the new name is what signs in from now on.
    assert (
        client.post(
            "/api/v1/auth/login", json={"username": new_name, "password": PASSWORD}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"username": account["username"], "password": PASSWORD},
        ).status_code
        == 401
    )


def test_no_token_is_issued_when_the_username_is_unchanged(client, account):
    response = client.patch(
        "/api/v1/users/me", json={"full_name": "Same Name"}, headers=account["headers"]
    )
    assert response.status_code == 200
    assert response.json()["access_token"] is None


def test_username_cannot_collide_with_another_account(client, account):
    taken = f"taken_{uuid.uuid4().hex[:8]}"
    client.post(
        "/api/v1/auth/signup",
        json={
            "username": taken,
            "email": f"{taken}@example.co.ke",
            "full_name": "Other",
            "account_type": "general",
            "password": PASSWORD,
        },
    )

    response = client.patch(
        "/api/v1/users/me", json={"username": taken}, headers=account["headers"]
    )
    assert response.status_code == 409
    # The original name must survive a rejected rename.
    assert client.get("/api/v1/users/me", headers=account["headers"]).json()["username"] == account["username"]


@pytest.mark.parametrize("bad", ["ab", "has spaces", "sym$bol", "x" * 51])
def test_invalid_usernames_are_rejected(client, account, bad):
    response = client.patch(
        "/api/v1/users/me", json={"username": bad}, headers=account["headers"]
    )
    assert response.status_code == 422
