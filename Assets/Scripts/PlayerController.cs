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
                    velocity.Set((float)myInput[0], (float)myInput[1]);
                    aim.Set((float)myInput[2], (float)myInput[3]);
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

                velocity.Set(move.x, move.y);
                aim.Set(aimInput.x, aimInput.y);

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
        if (velocity.magnitude > .1f)
        {
            rb.MovePosition(rb.position + (velocity * speed * DriverController.instance.fixedDeltaTime));//Time.fixedDeltaTime));
            rb.SetRotation(Vector2.SignedAngle(Vector2.right, velocity));
        }

        if (aim.magnitude > .1f)
        {
            barrel.rotation = Quaternion.Euler(0, 0, Vector2.SignedAngle(Vector2.right, aim));
        }
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
