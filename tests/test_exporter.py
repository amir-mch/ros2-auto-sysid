"""Unit tests for YAML parameter exporter and URDF updater."""

from pathlib import Path
import xml.etree.ElementTree as ET

from sysid_exporter.yaml_exporter import export_to_ros2_yaml, load_ros2_yaml
from sysid_exporter.urdf_updater import (
    create_default_f1tenth_urdf,
    update_urdf_inertial,
    update_urdf_dynamics,
    create_default_f1tenth_sdf,
    update_sdf_inertial,
    update_sdf_dynamics,
)


def test_yaml_export_and_load(tmp_path: Path):
    """Verify parameters export to valid ROS 2 YAML and reload accurately."""
    params = {
        "mass": 3.42,
        "yaw_inertia": 0.046,
        "c_f": 74.5,
        "c_r": 85.2,
    }
    yaml_file = tmp_path / "test_params.yaml"
    export_to_ros2_yaml(params, yaml_file, node_name="vehicle_node")

    assert yaml_file.exists()
    loaded = load_ros2_yaml(yaml_file, node_name="vehicle_node")

    for k, v in params.items():
        assert k in loaded
        assert abs(loaded[k] - v) < 1e-4


def test_urdf_inertial_updating(tmp_path: Path):
    """Verify URDF inertial mass and inertia values are updated correctly."""
    raw_urdf = tmp_path / "raw.urdf"
    cal_urdf = tmp_path / "calibrated.urdf"

    create_default_f1tenth_urdf(raw_urdf)
    assert raw_urdf.exists()

    calibrated_params = {
        "mass": 3.654,
        "yaw_inertia": 0.0521,
    }
    update_urdf_inertial(raw_urdf, cal_urdf, calibrated_params, target_link_name="base_link")
    assert cal_urdf.exists()

    tree = ET.parse(cal_urdf)
    root = tree.getroot()
    base_link = None
    for link in root.findall("link"):
        if link.get("name") == "base_link":
            base_link = link
            break

    assert base_link is not None
    mass_elem = base_link.find("inertial/mass")
    assert mass_elem is not None
    assert abs(float(mass_elem.get("value")) - 3.654) < 1e-4

    inertia_elem = base_link.find("inertial/inertia")
    assert inertia_elem is not None
    assert abs(float(inertia_elem.get("izz")) - 0.0521) < 1e-5


def test_urdf_missing_inertia_tag_updating(tmp_path: Path):
    """Regression test: verify link with <inertial> but no <inertia> child is handled without crash."""
    minimal_urdf = tmp_path / "minimal.urdf"
    out_urdf = tmp_path / "minimal_out.urdf"
    minimal_urdf.write_text("""<?xml version="1.0"?>
<robot name="minimal">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
    </inertial>
  </link>
</robot>
""")
    calibrated_params = {"mass": 3.5, "yaw_inertia": 0.048}
    update_urdf_inertial(minimal_urdf, out_urdf, calibrated_params, target_link_name="base_link")
    assert out_urdf.exists()

    tree = ET.parse(out_urdf)
    izz = tree.find(".//inertia").get("izz")
    assert abs(float(izz) - 0.048) < 1e-5


def test_urdf_joint_dynamics_updating(tmp_path: Path):
    """Verify URDF <dynamics damping="..." friction="..."/> is updated accurately."""
    raw_urdf = tmp_path / "f1tenth.urdf"
    out_urdf = tmp_path / "f1tenth_dyn.urdf"
    create_default_f1tenth_urdf(raw_urdf)

    update_urdf_dynamics(raw_urdf, out_urdf, joint_name="steering_joint", damping=0.085, friction=0.015)
    assert out_urdf.exists()

    tree = ET.parse(out_urdf)
    dyn = tree.find(".//joint[@name='steering_joint']/dynamics")
    assert dyn is not None
    assert abs(float(dyn.get("damping")) - 0.085) < 1e-5
    assert abs(float(dyn.get("friction")) - 0.015) < 1e-5


def test_sdf_inertial_and_dynamics_updating(tmp_path: Path):
    """Verify SDF <inertial> and <dynamics> tags are created and updated accurately."""
    raw_sdf = tmp_path / "f1tenth.sdf"
    out_sdf = tmp_path / "f1tenth_cal.sdf"
    create_default_f1tenth_sdf(raw_sdf)

    update_sdf_inertial(raw_sdf, out_sdf, {"mass": 3.75, "yaw_inertia": 0.055}, target_link_name="base_link")
    update_sdf_dynamics(out_sdf, out_sdf, joint_name="steering_joint", damping=0.075, friction=0.012)

    tree = ET.parse(out_sdf)
    mass_txt = tree.find(".//link[@name='base_link']/inertial/mass").text
    izz_txt = tree.find(".//link[@name='base_link']/inertial/inertia/izz").text
    assert abs(float(mass_txt) - 3.75) < 1e-4
    assert abs(float(izz_txt) - 0.055) < 1e-5

    damping_txt = tree.find(".//joint[@name='steering_joint']/axis/dynamics/damping").text
    friction_txt = tree.find(".//joint[@name='steering_joint']/axis/dynamics/friction").text
    assert abs(float(damping_txt) - 0.075) < 1e-5
    assert abs(float(friction_txt) - 0.012) < 1e-5
