from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET

import pytest

import nsai.advisor.client as advisor_client
from nsai.profile.auto import (
    AUTO_PROFILE_SHARDS,
    build_auto_profile,
    extract_freedom_scores,
    extract_spending_by_policy_area,
    freedom_score_weight,
    run_auto,
    spending_weight,
)


SAMPLE_NATION_XML = """
<NATION>
  <NAME>Oringrad</NAME>
  <FULLNAME>The Federation of Oringrad</FULLNAME>
  <MOTTO>Forward, together.</MOTTO>
  <CATEGORY>Inoffensive Centrist Democracy</CATEGORY>
  <GOVTDESC>A sprawling bureaucracy funds education and healthcare.</GOVTDESC>
  <SENSIBILITIES>Compassionate, Cultured</SENSIBILITIES>
  <FREEDOMSCORES>
    <CIVILRIGHTS>72</CIVILRIGHTS>
    <ECONOMY>44</ECONOMY>
    <POLITICALFREEDOM>61</POLITICALFREEDOM>
  </FREEDOMSCORES>
  <GOVT>
    <ADMINISTRATION>6.0</ADMINISTRATION>
    <DEFENCE>4.5</DEFENCE>
    <EDUCATION>21.0</EDUCATION>
    <ENVIRONMENT>8.0</ENVIRONMENT>
    <HEALTHCARE>18.5</HEALTHCARE>
    <COMMERCE>10.0</COMMERCE>
    <INTERNATIONALAID>1.0</INTERNATIONALAID>
    <LAWANDORDER>12.0</LAWANDORDER>
    <PUBLICTRANSPORT>5.0</PUBLICTRANSPORT>
    <SOCIALEQUALITY>4.0</SOCIALEQUALITY>
    <SPIRITUALITY>0.0</SPIRITUALITY>
    <WELFARE>10.0</WELFARE>
  </GOVT>
  <POLICIES>
    <POLICY>
      <NAME>Compulsory Vaccination</NAME>
      <CAT>Healthcare</CAT>
    </POLICY>
    <POLICY>
      <NAME>No Nukes</NAME>
      <CAT>Military</CAT>
    </POLICY>
  </POLICIES>
</NATION>
"""

CANTERWYN_LIKE_XML = """
<NATION>
  <NAME>Canterwyn</NAME>
  <FULLNAME>The Dominion of Canterwyn</FULLNAME>
  <MOTTO>For Stakeholders Universal</MOTTO>
  <CATEGORY>Corrupt Dictatorship</CATEGORY>
  <GOVTDESC>The medium-sized, corrupt, well-organized government juggles the competing demands of Law &amp; Order, Defense, and Education.</GOVTDESC>
  <SENSIBILITIES>Hard-nosed, Cynical, Humorless</SENSIBILITIES>
  <FREEDOMSCORES>
    <CIVILRIGHTS>39</CIVILRIGHTS>
    <ECONOMY>99</ECONOMY>
    <POLITICALFREEDOM>7</POLITICALFREEDOM>
  </FREEDOMSCORES>
  <GOVT>
    <DEFENCE>16.6</DEFENCE>
    <EDUCATION>13.2</EDUCATION>
    <ENVIRONMENT>5.5</ENVIRONMENT>
    <HEALTHCARE>6.7</HEALTHCARE>
    <COMMERCE>13.0</COMMERCE>
    <LAWANDORDER>18.2</LAWANDORDER>
    <PUBLICTRANSPORT>4.1</PUBLICTRANSPORT>
    <SOCIALEQUALITY>5.5</SOCIALEQUALITY>
    <WELFARE>6.5</WELFARE>
  </GOVT>
  <POLICIES>
    <POLICY><NAME>Autocracy</NAME></POLICY>
    <POLICY><NAME>State Press</NAME></POLICY>
    <POLICY><NAME>Pledge of Allegiance</NAME></POLICY>
  </POLICIES>
</NATION>
"""


def sample_nation_root() -> ET.Element:
    return ET.fromstring(SAMPLE_NATION_XML)


def canterwyn_like_root() -> ET.Element:
    return ET.fromstring(CANTERWYN_LIKE_XML)


def test_extract_freedom_scores() -> None:
    scores = extract_freedom_scores(sample_nation_root())

    assert scores == {
        'civil_rights': 72.0,
        'economy': 44.0,
        'political_freedom': 61.0,
    }


def test_extract_spending_maps_and_skips_zero() -> None:
    spending = extract_spending_by_policy_area(sample_nation_root())

    assert spending['education'] == 21.0
    assert spending['public_order'] == 12.0
    assert spending['civil_rights'] == 4.0
    assert 'religion' not in spending  # SPIRITUALITY is 0.0


def test_weight_helpers_clamp() -> None:
    assert freedom_score_weight(100.0) == 5
    assert freedom_score_weight(0.0) == -5
    assert freedom_score_weight(50.0) == 0
    assert spending_weight(0.5) == 1
    assert spending_weight(50.0) == 5


def test_build_auto_profile_shapes_profile_from_stats() -> None:
    profile = build_auto_profile(sample_nation_root())

    assert profile['nation_name'] == 'Oringrad'
    assert profile['profile_name'] == 'Auto Profile: Inoffensive Centrist Democracy'
    assert profile['enactment_mode'] == 'advise_only'
    assert profile['minimum_confidence_to_enact'] == 0.85

    # Education has the largest mapped spending share, so it leads priorities.
    assert profile['top_priorities'][0] == 'education'
    assert len(profile['top_priorities']) == 5
    assert set(profile['secondary_priorities']).isdisjoint(profile['top_priorities'])

    # Freedom scores override spending-derived weights for their areas.
    assert profile['scoring_weights']['civil_rights'] == 2
    assert profile['scoring_weights']['economy'] == -1
    assert profile['scoring_weights']['education'] == 5
    assert profile['scoring_weights']['weirdness'] == -2

    # High civil rights and political freedom add protective red lines.
    red_lines = '\n'.join(profile['red_lines'])
    assert 'civil rights' in red_lines
    assert 'political freedom' in red_lines
    assert 'strangle the existing strong economy' not in red_lines

    assert 'The Federation of Oringrad is a Inoffensive Centrist Democracy.' in (
        profile['roleplay_premise']
    )
    assert 'Compulsory Vaccination' in profile['roleplay_premise']
    assert 'compassionate, cultured' in profile['tone']

    meta = profile['auto_generated']
    assert meta['source'] == 'nationstates-public-stats'
    assert meta['freedom_scores']['civil_rights'] == 72.0


def test_build_auto_profile_continues_bad_current_trajectory() -> None:
    profile = build_auto_profile(canterwyn_like_root())

    assert profile['profile_name'] == 'Auto Profile: Corrupt Dictatorship'
    assert profile['governing_style'] == 'steady custodian of a corrupt dictatorship'
    assert profile['top_priorities'][:3] == ['public_order', 'military', 'education']
    assert profile['scoring_weights']['economy'] == 5
    assert profile['scoring_weights']['civil_rights'] == -1
    assert profile['scoring_weights']['political_freedom'] == -4

    profile_text = json.dumps(profile)
    assert 'Continue along the current national trajectory' in profile['national_vision']
    assert 'Govern the nation as it already is' in profile['custom_instruction']
    assert 'reform' not in profile_text.lower()
    assert 'repair' not in profile_text.lower()


def test_build_auto_profile_without_stats_uses_fallbacks() -> None:
    profile = build_auto_profile(ET.fromstring('<NATION/>'), nation_fallback='Testlandia')

    assert profile['nation_name'] == 'Testlandia'
    assert profile['top_priorities'] == ['economy', 'public_order']
    assert profile['secondary_priorities'] == []
    assert profile['scoring_weights'] == {'weirdness': -2}


def test_run_auto_writes_profile(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    requested: dict[str, object] = {}

    class FakeClient:
        def public_nation(self, nation, shards):
            requested['nation'] = nation
            requested['shards'] = shards
            return sample_nation_root()

    monkeypatch.setattr(
        advisor_client.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeClient()),
    )

    output = tmp_path / 'auto_profile.json'
    args = argparse.Namespace(nation='Oringrad', output=str(output), force=False)
    run_auto(args)

    assert requested['nation'] == 'Oringrad'
    assert requested['shards'] == AUTO_PROFILE_SHARDS

    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['nation_name'] == 'Oringrad'
    assert saved['enactment_mode'] == 'advise_only'
    output_text = capsys.readouterr().out
    assert 'Saved auto profile for Oringrad' in output_text
    assert f'nsai profile preview {output}' in output_text
    assert f'nsai nation set Oringrad --profile {output}' in output_text


def test_run_auto_refuses_overwrite_without_force(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))

    class FakeClient:
        def public_nation(self, nation, shards):
            return sample_nation_root()

    monkeypatch.setattr(
        advisor_client.NationStatesClient,
        'from_env',
        classmethod(lambda cls, nation_config=None: FakeClient()),
    )

    output = tmp_path / 'auto_profile.json'
    output.write_text('{}', encoding='utf-8')

    args = argparse.Namespace(nation='Oringrad', output=str(output), force=False)
    with pytest.raises(SystemExit, match='use --force'):
        run_auto(args)

    args = argparse.Namespace(nation='Oringrad', output=str(output), force=True)
    run_auto(args)
    assert json.loads(output.read_text(encoding='utf-8'))['nation_name'] == 'Oringrad'


def test_run_auto_rejects_profile_path_before_api_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config-home'))
    profile_path = tmp_path / 'canterwyn_auto_profile.json'
    profile_path.write_text('{}', encoding='utf-8')

    def fail_from_env(cls, nation_config=None):  # noqa: ANN001
        raise AssertionError('profile path should be rejected before API client setup')

    monkeypatch.setattr(
        advisor_client.NationStatesClient,
        'from_env',
        classmethod(fail_from_env),
    )

    args = argparse.Namespace(nation=str(profile_path), output=None, force=False)
    with pytest.raises(SystemExit, match='looks like an existing profile file'):
        run_auto(args)


def test_run_auto_requires_nation(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv('NS_NATION', raising=False)
    monkeypatch.setenv('NSAI_CONFIG_HOME', str(tmp_path / 'config'))

    args = argparse.Namespace(nation=None, output=None, force=False)
    with pytest.raises(SystemExit, match='Provide a nation name'):
        run_auto(args)
