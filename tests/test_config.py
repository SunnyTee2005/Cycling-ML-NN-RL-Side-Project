from cycling_coach import config


def test_system_mass_adds_rider_kit_and_bike():
    expected = config.RIDER_MASS_KG + config.KIT_MASS_KG + 8.0
    assert config.system_mass_kg("TCR BLACK") == expected


def test_data_dir_lives_inside_project():
    assert config.DATA_DIR.parent == config.PROJECT_ROOT
