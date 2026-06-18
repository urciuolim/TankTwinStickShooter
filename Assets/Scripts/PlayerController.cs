using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;
using System;
using UnityEngine.Tilemaps;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Users;

public class PlayerController : MonoBehaviour
{
    private float speed = 3;
    private Rigidbody2D rb;
    [HideInInspector]
    public Vector2 velocity;
    [HideInInspector]
    public Vector2 aim;

    // Smooth-movement tuning (CTO play-test dials these in the Inspector).
    // The action [velX, velY, aimX, aimY] is the DESIRED direction; these control how
    // fast the tank's ACTUAL velocity (`velocity`) and heading (`aim`) ramp toward it.
    [Header("Smooth movement (tune for feel)")]
    [Tooltip("How fast actual velocity ramps toward desired, in direction-units/sec " +
             "(velocity is the normalized stick/action direction, ~0..1). Lower = more " +
             "inertia/slide; higher = snappier. ~999 reproduces the old instant snap.")]
    [SerializeField]
    private float moveAcceleration = 6f;
    [Tooltip("Max turn rate of the tank body + barrel heading, degrees/sec. Lower = " +
             "slower, heavier turning; higher = quicker. ~9999 reproduces instant aim.")]
    [SerializeField]
    private float turnSpeedDegPerSec = 360f;

    // The latest DESIRED direction off the action/input (RL action space is unchanged).
    // FixedUpdate smooths the ACTUAL `velocity`/`aim` toward these.
    private Vector2 desiredVelocity;
    private Vector2 desiredAim;
    // Actual body/barrel heading in degrees; smoothed toward the desired aim angle.
    private float currentAimAngle;
    private Transform barrel;
    public GameObject bulletPrefab;
    private Transform firePoint;
    private float triggerThreshold = .5f;
    private float reloadTime = .33f;
    [HideInInspector]
    public bool canShoot = true;
    private int reloadSteps;
    private int reloadCountdown;
    private bool randomStart = true;
    private float xSpawnLim = .5f;
    private float ySpawnLim = 3f;

    private bool keyboard = false;

    public Color myColor;
    [HideInInspector]
    public float maxHealth = 1;
    [HideInInspector]
    public float health;
    public int playerID;
    private string horizontal, vertical, r_horizontal, r_vertical, trigger;

    // New Input System (human branch only). The asset is a template; each human tank
    // gets its OWN clone with explicit device pairing + a binding-group mask so two
    // identical pads (or the two shared-keyboard clusters) never cross-assign.
    [SerializeField]
    private InputActionAsset tankControls;
    private InputActionAsset humanControls;
    private InputAction moveAction, aimAction, fireAction;

    [HideInInspector]
    public bool AI = true;

    [HideInInspector]
    public bool playing = false;

    [HideInInspector]
    public List<GameObject> bullets;

    private void Awake()
    {
        JObject config = DriverController.instance.config;
        if (config["player_speed"] != null)
            speed = config["player_speed"].Value<float>();
        if (config["player_triggerThreshold"] != null)
            triggerThreshold = config["player_triggerThreshold"].Value<float>();
        if (config["player_reloadTime"] != null)
            reloadTime = config["player_reloadTime"].Value<float>();
        if (config["player_maxHealth"] != null)
            maxHealth = config["player_maxHealth"].Value<float>();
        if (config["player" + playerID + "_ai"] != null)
            AI = config["player" + playerID + "_ai"].Value<bool>();
        if (config["player" + playerID + "_keyboard"] != null)
            keyboard = config["player" + playerID + "_keyboard"].Value<bool>();
        if (config["player_randomStart"] != null)
            randomStart = config["player_randomStart"].Value<bool>();
        if (config["player_x_spawn_lim"] != null)
            xSpawnLim = config["player_x_spawn_lim"].Value<float>();
        if (config["player_y_spawn_lim"] != null)
            ySpawnLim = config["player_y_spawn_lim"].Value<float>();
        if (config["player_moveAcceleration"] != null)
            moveAcceleration = config["player_moveAcceleration"].Value<float>();
        if (config["player_turnSpeedDegPerSec"] != null)
            turnSpeedDegPerSec = config["player_turnSpeedDegPerSec"].Value<float>();
        if (DriverController.instance.verbose)
        {
            Debug.Log("Player " + playerID + " speed set to " + speed);
            Debug.Log("Player " + playerID + " trigger threshold set to " + triggerThreshold);
            Debug.Log("Player " + playerID + " reload time set to " + reloadTime);
            Debug.Log("Player " + playerID + " max health set to " + maxHealth);
            Debug.Log("Player " + playerID + " random start set to " + randomStart);
        }
    }

    // Start is called before the first frame update
    private void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        velocity = new Vector2();
        aim = new Vector2();
        desiredVelocity = Vector2.zero;
        desiredAim = Vector2.zero;
        currentAimAngle = rb.rotation;
        barrel = transform.Find("Barrel");
        firePoint = barrel.Find("FirePoint");
        canShoot = true;
        health = maxHealth;

        transform.Find("Body").GetComponent<SpriteRenderer>().color = myColor;
        barrel.Find("BarrelSprite").gameObject.GetComponent<SpriteRenderer>().color = myColor;
        horizontal = "HorizontalJoy" + playerID;
        vertical = "VerticalJoy" + playerID;
        r_horizontal = "R_HorizontalJoy" + playerID;
        r_vertical = "R_VerticalJoy" + playerID;
        trigger = "TriggerJoy" + playerID;

        if (!AI)
        {
            SetupHumanInput();
        }

        bullets = new List<GameObject>();
        reloadSteps = (int)Math.Ceiling(reloadTime / DriverController.instance.fixedDeltaTime);
        reloadCountdown = 0;

        if (randomStart)
        {
            RandomStart();
        }
        else
        {
            if (DriverController.instance.flip)
            {
                transform.position = new Vector3(-transform.position.x, -transform.position.y);
            }
            transform.position = new Vector3(transform.position.x + UnityEngine.Random.Range(-xSpawnLim, xSpawnLim),
                                            transform.position.y + UnityEngine.Random.Range(-ySpawnLim, ySpawnLim));
        }
    }

    private void RandomStart()
    {
        Tilemap floor = GameObject.Find("Floor").GetComponent<Tilemap>();
        float minX = floor.cellBounds.xMin + .5f;
        float maxX = floor.cellBounds.xMax - .5f;
        float minY = floor.cellBounds.yMin + .5f;
        float maxY = floor.cellBounds.yMax - .5f;

        float newX = UnityEngine.Random.Range(minX, maxX);
        float newY = UnityEngine.Random.Range(minY, maxY);

        transform.position = new Vector3(newX, newY);
    }

    // Update is called once per frame
    private void Update()
    {
        //GetInput();
        //Debug.Log("Update FPS: " + (1 / Time.deltaTime));
    }

    // Build this human tank's input from the New Input System: clone the shared
    // InputActions template, pin it to ONE device, and mask it to ONE binding group.
    // The "keyboard" config flag selects keyboard-scheme vs pad for this tank; playerID
    // both picks which pad (1 -> first pad, 2 -> second pad, mirroring the legacy
    // JoyN axes) and which shared-keyboard cluster (1 -> left/WASD, 2 -> right/arrows).
    private void SetupHumanInput()
    {
        if (tankControls == null)
        {
            Debug.LogError("Player " + playerID + " is human but no InputActionAsset is assigned; falling back to no input.");
            return;
        }

        // A per-player clone so device pairing / binding masks don't leak between tanks.
        humanControls = Instantiate(tankControls);
        InputActionMap tankMap = humanControls.FindActionMap("Tank", true);

        if (keyboard)
        {
            // Both keyboard players share the SAME physical keyboard; they are kept
            // disjoint by binding group, not by device. playerID 1 -> left cluster,
            // playerID 2 -> right cluster.
            string scheme = (playerID == 1) ? "KeyboardLeft" : "KeyboardRight";
            tankMap.bindingMask = InputBinding.MaskByGroup(scheme);
            if (Keyboard.current != null)
                tankMap.devices = new InputDevice[] { Keyboard.current };
        }
        else
        {
            // Pin this tank to a specific pad so two identical pads don't cross-assign:
            // playerID 1 -> Gamepad.all[0], playerID 2 -> Gamepad.all[1] (matches the
            // legacy Joy1/Joy2 mapping). If the expected pad isn't present we leave the
            // map device-unrestricted as a graceful fallback (any pad drives the tank).
            tankMap.bindingMask = InputBinding.MaskByGroup("Gamepad");
            int padIndex = playerID - 1;
            if (Gamepad.all.Count > padIndex)
                tankMap.devices = new InputDevice[] { Gamepad.all[padIndex] };
            else
                Debug.LogWarning("Player " + playerID + " expected pad index " + padIndex +
                                 " but only " + Gamepad.all.Count + " pad(s) present; using any pad.");
        }

        moveAction = tankMap.FindAction("Move", true);
        aimAction = tankMap.FindAction("Aim", true);
        fireAction = tankMap.FindAction("Fire", true);
        tankMap.Enable();
    }

    private void OnDestroy()
    {
        if (humanControls != null)
        {
            humanControls.Disable();
            Destroy(humanControls);
        }
    }

    private void GetInput()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
            Application.Quit();

        if (playing)
        {
            if (AI)
            {
                if (DriverController.instance.actions != null)
                {
                    JArray myInput = DriverController.instance.actions[playerID.ToString()] as JArray;
                    // DESIRED direction off the socket; FixedUpdate smooths the ACTUAL
                    // velocity/aim toward it (RL action space + path unchanged).
                    desiredVelocity.Set((float)myInput[0], (float)myInput[1]);
                    desiredAim.Set((float)myInput[2], (float)myInput[3]);
                    if ((float)myInput[4] > triggerThreshold && canShoot)
                    {
                        Shoot();
                    }
                }
            }
            else
            {
                // Human input via the New Input System. Move/Aim/Fire come from this
                // tank's per-player action clone (one paired device, one binding-group
                // mask: pad left/right sticks + right trigger, OR a shared-keyboard
                // cluster with key-aim). Velocity/aim/trigger are applied exactly as
                // before; only the source of the values changed.
                Vector2 move = Vector2.zero;
                Vector2 aimInput = Vector2.zero;
                float triggerValue = 0f;
                if (moveAction != null)
                    move = moveAction.ReadValue<Vector2>();
                if (aimAction != null)
                    aimInput = aimAction.ReadValue<Vector2>();
                if (fireAction != null)
                    triggerValue = fireAction.ReadValue<float>();

                // DESIRED direction from the pad/keyboard; FixedUpdate smooths the ACTUAL
                // velocity/aim toward it, identically to the AI branch.
                desiredVelocity.Set(move.x, move.y);
                desiredAim.Set(aimInput.x, aimInput.y);

                bool triggerPressed = triggerValue > triggerThreshold;
                if (triggerPressed && canShoot)
                {
                    Shoot();
                }
            }
        }
    }

    private void Shoot()
    {
        canShoot = false;
        GameController.instance.UpdateReloading(playerID, canShoot);
        GameObject b = Instantiate(bulletPrefab, firePoint.position, barrel.rotation);
        b.GetComponent<SpriteRenderer>().color = myColor;
        b.GetComponent<BulletController>().shooter = this;
        bullets.Add(b);
        reloadCountdown = reloadSteps;
        //StartCoroutine(Reload());
    }

    public void DestroyBullet(GameObject bullet)
    {
        bullets.Remove(bullet);
        Destroy(bullet);
    }

    IEnumerator Reload()
    {
        yield return new WaitForSeconds(reloadTime);
        canShoot = true;
        GameController.instance.UpdateReloading(playerID, canShoot);
    }
    private void FixedUpdate()
    {
        float dt = DriverController.instance.fixedDeltaTime;//Time.fixedDeltaTime;

        // --- Movement smoothing (acceleration / inertia) ---------------------
        // Ramp the ACTUAL velocity toward the DESIRED direction so commanding a
        // direction accelerates and releasing decelerates (momentum, not snap).
        // `velocity` stays direction-space (~0..1) and is multiplied by `speed` below,
        // matching the original semantics + the state-vector layout.
        velocity = Vector2.MoveTowards(velocity, desiredVelocity, moveAcceleration * dt);

        if (velocity.magnitude > .1f)
        {
            rb.MovePosition(rb.position + (velocity * speed * dt));
            // Body faces the actual (smoothed) movement direction, as before.
            rb.SetRotation(Vector2.SignedAngle(Vector2.right, velocity));
        }

        // --- Aim smoothing (max angular rate) --------------------------------
        // Only chase a new heading when a meaningful aim is commanded (mirrors the
        // old `aim.magnitude > .1f` gate, so no aim input doesn't drag toward angle 0).
        if (desiredAim.magnitude > .1f)
        {
            // Actively aiming: smooth the independent barrel heading toward the
            // commanded world target and force the barrel's world rotation.
            float targetAngle = Vector2.SignedAngle(Vector2.right, desiredAim);
            currentAimAngle = Mathf.MoveTowardsAngle(currentAimAngle, targetAngle, turnSpeedDegPerSec * dt);
            barrel.rotation = Quaternion.Euler(0, 0, currentAimAngle);
        }
        else
        {
            // Not aiming: leave the barrel's world rotation alone so it rides with
            // the body (the body's SetRotation above carries this child transform).
            // Sync currentAimAngle to the barrel's ACTUAL heading so (a) smoothing
            // resumes from the right angle next time we aim and (b) the reported
            // `aim` below reflects the barrel's true world heading.
            currentAimAngle = barrel.rotation.eulerAngles.z;
        }
        // Report the ACTUAL heading as a unit vector so UpdateState's `aim.x/aim.y`
        // is the real barrel world heading (smoothed aim, or the body when idle),
        // not the raw command.
        float rad = currentAimAngle * Mathf.Deg2Rad;
        aim.Set(Mathf.Cos(rad), Mathf.Sin(rad));

        GetInput();
        if (!canShoot)
        {
            reloadCountdown--;
            if (reloadCountdown <= 0)
            {
                canShoot = true;
                GameController.instance.UpdateReloading(playerID, canShoot);
            }
        }
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.tag == "Bullet" && collision.gameObject.GetComponent<BulletController>().shooter != this)
        {
            //Debug.Log(playerID + " got shot");
            health -= collision.gameObject.GetComponent<BulletController>().damage;
            GameController.instance.UpdateHealth(playerID);
            if (health <= 0)
            {
                Debug.Log("Player " + playerID + " died");
                Die();
            }
        }
    }

    private void Die()
    {
        Destroy(gameObject);
        GameController.instance.CheckGameEnd();
    }

    public void DeactivateBullets()
    {
        foreach (var bullet in bullets)
        {
            bullet.SetActive(false);
        }
    }
}
