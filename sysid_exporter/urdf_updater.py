"""URDF and SDF physics parameter injector.

Updates inertial properties (mass, inertia matrix) and joint dynamics
(<dynamics> damping, friction) in robot description files (URDF and SDF)
from system identification results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import xml.etree.ElementTree as ET


def update_urdf_inertial(
    urdf_input_path: str | Path,
    urdf_output_path: str | Path,
    calibrated_params: Dict[str, float],
    target_link_name: str = "base_link",
) -> Path:
    """Update <inertial> properties of target_link in URDF from calibrated params.

    Args:
        urdf_input_path: Source URDF template file.
        urdf_output_path: Target path to write calibrated URDF.
        calibrated_params: Dictionary containing 'mass' and 'yaw_inertia' (I_z).
        target_link_name: Name of chassis or vehicle body link.

    Returns:
        Path to the modified URDF.
    """
    in_path = Path(urdf_input_path)
    out_path = Path(urdf_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_path)
    root = tree.getroot()

    link_found = False
    for link in root.findall("link"):
        if link.get("name") == target_link_name:
            link_found = True
            inertial = link.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(link, "inertial")

            # Update mass
            if "mass" in calibrated_params:
                mass_elem = inertial.find("mass")
                if mass_elem is None:
                    mass_elem = ET.SubElement(inertial, "mass")
                mass_elem.set("value", f"{calibrated_params['mass']:.5f}")

            # Update yaw moment of inertia (izz)
            if "yaw_inertia" in calibrated_params:
                inertia_elem = inertial.find("inertia")
                if inertia_elem is None:
                    inertia_elem = ET.SubElement(inertial, "inertia")
                    inertia_elem.set("ixx", "0.01")
                    inertia_elem.set("ixy", "0.0")
                    inertia_elem.set("ixz", "0.0")
                    inertia_elem.set("iyy", "0.02")
                    inertia_elem.set("iyz", "0.0")

                new_izz = calibrated_params["yaw_inertia"]
                inertia_elem.set("izz", f"{new_izz:.6f}")

    if not link_found:
        raise ValueError(f"Link '{target_link_name}' not found in URDF '{urdf_input_path}'")

    # Write formatted XML
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def update_urdf_dynamics(
    urdf_input_path: str | Path,
    urdf_output_path: str | Path,
    joint_name: str,
    damping: Optional[float] = None,
    friction: Optional[float] = None,
) -> Path:
    """Update <dynamics> (damping and friction) attributes of a target joint in URDF.

    Args:
        urdf_input_path: Source URDF file.
        urdf_output_path: Destination path for updated URDF.
        joint_name: Name of target joint (e.g. 'steering_joint').
        damping: Calibrated viscous joint damping [N*m*s/rad].
        friction: Calibrated static friction [N*m].

    Returns:
        Path to updated URDF file.
    """
    in_path = Path(urdf_input_path)
    out_path = Path(urdf_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_path)
    root = tree.getroot()

    joint_found = False
    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            joint_found = True
            dyn = joint.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(joint, "dynamics")
            if damping is not None:
                dyn.set("damping", f"{damping:.6f}")
            if friction is not None:
                dyn.set("friction", f"{friction:.6f}")

    if not joint_found:
        raise ValueError(f"Joint '{joint_name}' not found in URDF '{urdf_input_path}'")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def update_sdf_inertial(
    sdf_input_path: str | Path,
    sdf_output_path: str | Path,
    calibrated_params: Dict[str, float],
    target_link_name: str = "base_link",
) -> Path:
    """Update <inertial> properties of target_link in an SDF model.

    In SDF, values are stored as child text elements rather than XML attributes:
    <inertial>
      <mass>3.4</mass>
      <inertia>
        <izz>0.045</izz>
      </inertia>
    </inertial>

    Args:
        sdf_input_path: Source SDF model file.
        sdf_output_path: Destination path for calibrated SDF.
        calibrated_params: Dictionary with 'mass' and 'yaw_inertia'.
        target_link_name: Target link name.

    Returns:
        Path to modified SDF file.
    """
    in_path = Path(sdf_input_path)
    out_path = Path(sdf_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_path)
    root = tree.getroot()

    link_found = False
    for link in root.iter("link"):
        if link.get("name") == target_link_name:
            link_found = True
            inertial = link.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(link, "inertial")

            if "mass" in calibrated_params:
                mass_elem = inertial.find("mass")
                if mass_elem is None:
                    mass_elem = ET.SubElement(inertial, "mass")
                mass_elem.text = f"{calibrated_params['mass']:.5f}"

            if "yaw_inertia" in calibrated_params:
                inertia_elem = inertial.find("inertia")
                if inertia_elem is None:
                    inertia_elem = ET.SubElement(inertial, "inertia")
                    for axis in ["ixx", "ixy", "ixz", "iyy", "iyz"]:
                        sub = ET.SubElement(inertia_elem, axis)
                        sub.text = "0.01" if axis in ["ixx", "iyy"] else "0.0"

                izz_elem = inertia_elem.find("izz")
                if izz_elem is None:
                    izz_elem = ET.SubElement(inertia_elem, "izz")
                izz_elem.text = f"{calibrated_params['yaw_inertia']:.6f}"

    if not link_found:
        raise ValueError(f"Link '{target_link_name}' not found in SDF '{sdf_input_path}'")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def update_sdf_dynamics(
    sdf_input_path: str | Path,
    sdf_output_path: str | Path,
    joint_name: str,
    damping: Optional[float] = None,
    friction: Optional[float] = None,
) -> Path:
    """Update <dynamics> (damping and friction) of a joint in an SDF model."""
    in_path = Path(sdf_input_path)
    out_path = Path(sdf_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_path)
    root = tree.getroot()

    joint_found = False
    for joint in root.iter("joint"):
        if joint.get("name") == joint_name:
            joint_found = True
            axis = joint.find("axis")
            if axis is None:
                axis = ET.SubElement(joint, "axis")
            dyn = axis.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(axis, "dynamics")

            if damping is not None:
                d_elem = dyn.find("damping")
                if d_elem is None:
                    d_elem = ET.SubElement(dyn, "damping")
                d_elem.text = f"{damping:.6f}"

            if friction is not None:
                f_elem = dyn.find("friction")
                if f_elem is None:
                    f_elem = ET.SubElement(dyn, "friction")
                f_elem.text = f"{friction:.6f}"

    if not joint_found:
        raise ValueError(f"Joint '{joint_name}' not found in SDF '{sdf_input_path}'")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def create_default_f1tenth_urdf(output_path: str | Path) -> Path:
    """Generate a minimal valid F1TENTH URDF for calibration and testing."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    urdf_content = """<?xml version="1.0"?>
<robot name="f1tenth_vehicle">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="3.4000"/>
      <inertia ixx="0.0150" ixy="0.0" ixz="0.0" iyy="0.0350" iyz="0.0" izz="0.0450"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.50 0.20 0.12"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.50 0.20 0.12"/>
      </geometry>
    </collision>
  </link>
  <link name="steering_link"/>
  <joint name="steering_joint" type="revolute">
    <parent link="base_link"/>
    <child link="steering_link"/>
    <origin xyz="0.16 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.4" upper="0.4" effort="10.0" velocity="8.0"/>
    <dynamics damping="0.0500" friction="0.0100"/>
  </joint>
</robot>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(urdf_content)
    return path


def create_default_f1tenth_sdf(output_path: str | Path) -> Path:
    """Generate a minimal valid F1TENTH SDF model for Gazebo/Ignition testing."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sdf_content = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="f1tenth_vehicle">
    <link name="base_link">
      <inertial>
        <mass>3.4000</mass>
        <inertia>
          <ixx>0.0150</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.0350</iyy>
          <iyz>0.0</iyz>
          <izz>0.0450</izz>
        </inertia>
      </inertial>
    </link>
    <link name="steering_link"/>
    <joint name="steering_joint" type="revolute">
      <parent>base_link</parent>
      <child>steering_link</child>
      <axis>
        <xyz>0 0 1</xyz>
        <dynamics>
          <damping>0.0500</damping>
          <friction>0.0100</friction>
        </dynamics>
      </axis>
    </joint>
  </model>
</sdf>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(sdf_content)
    return path
