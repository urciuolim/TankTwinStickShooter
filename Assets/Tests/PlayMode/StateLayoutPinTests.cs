using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace TankTwinStickShooter.PlayModeTests
{
    // RL-SEAM REGRESSION GUARD (M0 cleanup B3).
    //
    // Pins the CURRENT GameController.UpdateState() float[52] observation layout EXACTLY,
    // WITHOUT modifying UpdateState or any RL-seam production code (per CLAUDE.md "Don't
    // touch the RL seam without sign-off"). We construct known tank/bullet state on INACTIVE
    // GameObjects (so no Awake/Start lifecycle runs and no config is needed), point a
    // GameController's private `tanks` field at it via reflection, invoke the private
    // UpdateState() via reflection, and assert the 52-vector that lands on
    // DriverController.instance.state. All reflection is test-side; production is untouched.
    //
    // Pinned contract (must match GameController.cs UpdateState):
    //   slots  0- 5 : P1 (tanks[0][0]) posX, posY, velX, velY, aimX, aimY
    //   slots  6-25 : P1's bullets, 4 floats each (posX, posY, velX*speed, velY*speed), up to 5
    //   slots 26-31 : P2 (tanks[1][0]) posX, posY, velX, velY, aimX, aimY
    //   slots 32-51 : P2's bullets, 4 floats each, up to 5
    //   every slot defaults to -100 (fill for absent bullets)
    //   bullet velocity is stored as BulletController.velocity.{x,y} * BulletController.speed
    public class StateLayoutPinTests
    {
        private readonly List<GameObject> spawned = new List<GameObject>();

        // Saved so we can restore DriverController.instance after the test.
        private DriverController previousInstance;

        [SetUp]
        public void SetUp()
        {
            previousInstance = DriverController.instance;
        }

        [TearDown]
        public void TearDown()
        {
            DriverController.instance = previousInstance;
            foreach (var go in spawned)
            {
                if (go != null)
                    Object.DestroyImmediate(go);
            }
            spawned.Clear();
        }

        // Create an INACTIVE GameObject + component so no Unity lifecycle method runs.
        private T NewInactiveComponent<T>(string name) where T : Component
        {
            var go = new GameObject(name);
            go.SetActive(false);
            spawned.Add(go);
            return go.AddComponent<T>();
        }

        private GameObject NewBullet(Vector2 pos, Vector2 vel, float speed)
        {
            var bc = NewInactiveComponent<BulletController>("Bullet");
            bc.transform.position = new Vector3(pos.x, pos.y, 0f);
            bc.velocity = new Vector3(vel.x, vel.y, 0f);
            bc.speed = speed;
            return bc.gameObject;
        }

        private PlayerController NewTank(Vector2 pos, Vector2 vel, Vector2 aim, List<GameObject> bullets)
        {
            var pc = NewInactiveComponent<PlayerController>("Tank");
            pc.transform.position = new Vector3(pos.x, pos.y, 0f);
            pc.velocity = vel;
            pc.aim = aim;
            pc.bullets = bullets;
            return pc;
        }

        private float[] InvokeUpdateState(PlayerController p1, PlayerController p2)
        {
            // A DriverController.instance must exist: UpdateState ends with
            // `DriverController.instance.state = state;`. Use an inactive instance so Awake
            // doesn't run; set the static singleton directly.
            var driver = NewInactiveComponent<DriverController>("Driver");
            DriverController.instance = driver;
            driver.state = null;

            var gc = NewInactiveComponent<GameController>("GameController");

            // tanks is private PlayerController[][]; UpdateState reads tanks[0][0] and tanks[1][0].
            var tanks = new PlayerController[2][];
            tanks[0] = new PlayerController[] { p1 };
            tanks[1] = new PlayerController[] { p2 };

            FieldInfo tanksField = typeof(GameController)
                .GetField("tanks", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(tanksField, "GameController.tanks field not found (layout/refactor drift?)");
            tanksField.SetValue(gc, tanks);

            MethodInfo updateState = typeof(GameController)
                .GetMethod("UpdateState", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(updateState, "GameController.UpdateState() not found (RL-seam drift?)");
            updateState.Invoke(gc, null);

            Assert.IsNotNull(driver.state, "UpdateState did not assign DriverController.instance.state");
            JArray arr = (JArray)driver.state["state"];
            Assert.IsNotNull(arr, "state JObject has no 'state' array");
            Assert.AreEqual(52, arr.Count, "state vector must be exactly 52 floats");

            var result = new float[52];
            for (int i = 0; i < 52; i++)
                result[i] = arr[i].Value<float>();
            return result;
        }

        [Test]
        public void UpdateState_PinsFullLayout_WithPartialBullets()
        {
            // P1: distinct kinematics + 2 bullets (so slots 6-13 populated, 14-25 stay -100).
            var p1b0 = NewBullet(new Vector2(11f, 12f), new Vector2(0.5f, -0.25f), 4f);
            var p1b1 = NewBullet(new Vector2(13f, 14f), new Vector2(-1f, 2f), 3f);
            var p1 = NewTank(
                pos: new Vector2(1f, 2f),
                vel: new Vector2(3f, 4f),
                aim: new Vector2(5f, 6f),
                bullets: new List<GameObject> { p1b0, p1b1 });

            // P2: distinct kinematics + 1 bullet (slot 32-35 populated, 36-51 stay -100).
            var p2b0 = NewBullet(new Vector2(21f, 22f), new Vector2(2f, 3f), 5f);
            var p2 = NewTank(
                pos: new Vector2(7f, 8f),
                vel: new Vector2(9f, 10f),
                aim: new Vector2(-1f, -2f),
                bullets: new List<GameObject> { p2b0 });

            float[] s = InvokeUpdateState(p1, p2);

            // P1 kinematics, slots 0-5.
            Assert.AreEqual(1f, s[0], "s[0] P1 posX");
            Assert.AreEqual(2f, s[1], "s[1] P1 posY");
            Assert.AreEqual(3f, s[2], "s[2] P1 velX");
            Assert.AreEqual(4f, s[3], "s[3] P1 velY");
            Assert.AreEqual(5f, s[4], "s[4] P1 aimX");
            Assert.AreEqual(6f, s[5], "s[5] P1 aimY");

            // P1 bullet 0, slots 6-9 (velocity scaled by speed).
            Assert.AreEqual(11f, s[6], "s[6] P1 b0 posX");
            Assert.AreEqual(12f, s[7], "s[7] P1 b0 posY");
            Assert.AreEqual(0.5f * 4f, s[8], 1e-4f, "s[8] P1 b0 velX*speed");
            Assert.AreEqual(-0.25f * 4f, s[9], 1e-4f, "s[9] P1 b0 velY*speed");

            // P1 bullet 1, slots 10-13.
            Assert.AreEqual(13f, s[10], "s[10] P1 b1 posX");
            Assert.AreEqual(14f, s[11], "s[11] P1 b1 posY");
            Assert.AreEqual(-1f * 3f, s[12], 1e-4f, "s[12] P1 b1 velX*speed");
            Assert.AreEqual(2f * 3f, s[13], 1e-4f, "s[13] P1 b1 velY*speed");

            // P1 absent bullets 2-4 -> slots 14-25 all -100.
            for (int i = 14; i <= 25; i++)
                Assert.AreEqual(-100f, s[i], "s[" + i + "] absent P1 bullet fill");

            // P2 kinematics, slots 26-31.
            Assert.AreEqual(7f, s[26], "s[26] P2 posX");
            Assert.AreEqual(8f, s[27], "s[27] P2 posY");
            Assert.AreEqual(9f, s[28], "s[28] P2 velX");
            Assert.AreEqual(10f, s[29], "s[29] P2 velY");
            Assert.AreEqual(-1f, s[30], "s[30] P2 aimX");
            Assert.AreEqual(-2f, s[31], "s[31] P2 aimY");

            // P2 bullet 0, slots 32-35.
            Assert.AreEqual(21f, s[32], "s[32] P2 b0 posX");
            Assert.AreEqual(22f, s[33], "s[33] P2 b0 posY");
            Assert.AreEqual(2f * 5f, s[34], 1e-4f, "s[34] P2 b0 velX*speed");
            Assert.AreEqual(3f * 5f, s[35], 1e-4f, "s[35] P2 b0 velY*speed");

            // P2 absent bullets 1-4 -> slots 36-51 all -100.
            for (int i = 36; i <= 51; i++)
                Assert.AreEqual(-100f, s[i], "s[" + i + "] absent P2 bullet fill");
        }

        [Test]
        public void UpdateState_NoBullets_AllBulletSlotsAreMinus100()
        {
            var p1 = NewTank(new Vector2(1f, 1f), new Vector2(2f, 2f), new Vector2(3f, 3f),
                new List<GameObject>());
            var p2 = NewTank(new Vector2(4f, 4f), new Vector2(5f, 5f), new Vector2(6f, 6f),
                new List<GameObject>());

            float[] s = InvokeUpdateState(p1, p2);

            // Kinematics still land; every bullet slot is the -100 fill.
            Assert.AreEqual(1f, s[0]);
            Assert.AreEqual(4f, s[26]);
            for (int i = 6; i <= 25; i++)
                Assert.AreEqual(-100f, s[i], "P1 bullet slot " + i + " must be -100 with no bullets");
            for (int i = 32; i <= 51; i++)
                Assert.AreEqual(-100f, s[i], "P2 bullet slot " + i + " must be -100 with no bullets");
        }

        [Test]
        public void UpdateState_FiveBulletsEach_FillEveryBulletSlot()
        {
            // 5 bullets per tank fully populates 6-25 and 32-51 (no -100 fill left in bullet range).
            var p1Bullets = new List<GameObject>();
            for (int k = 0; k < 5; k++)
                p1Bullets.Add(NewBullet(new Vector2(100 + k, 200 + k), new Vector2(1f, 1f), 2f));
            var p2Bullets = new List<GameObject>();
            for (int k = 0; k < 5; k++)
                p2Bullets.Add(NewBullet(new Vector2(300 + k, 400 + k), new Vector2(1f, 1f), 2f));

            var p1 = NewTank(Vector2.zero, Vector2.zero, Vector2.zero, p1Bullets);
            var p2 = NewTank(Vector2.zero, Vector2.zero, Vector2.zero, p2Bullets);

            float[] s = InvokeUpdateState(p1, p2);

            // Spot-check the last bullet of each tank lands at the final slots.
            Assert.AreEqual(104f, s[22], "P1 b4 posX at slot 22");
            Assert.AreEqual(204f, s[23], "P1 b4 posY at slot 23");
            Assert.AreEqual(304f, s[48], "P2 b4 posX at slot 48");
            Assert.AreEqual(404f, s[49], "P2 b4 posY at slot 49");

            // No -100 fill remains anywhere in the bullet ranges.
            for (int i = 6; i <= 25; i++)
                Assert.AreNotEqual(-100f, s[i], "P1 bullet slot " + i + " should be filled");
            for (int i = 32; i <= 51; i++)
                Assert.AreNotEqual(-100f, s[i], "P2 bullet slot " + i + " should be filled");
        }
    }
}
