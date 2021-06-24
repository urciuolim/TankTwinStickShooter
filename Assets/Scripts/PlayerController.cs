using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;
using System;
using UnityEngine.Tilemaps;

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
                float vX = 0f;
                float vY = 0f;
                if (keyboard)
                {
                    if (Input.GetKey(KeyCode.D) && !Input.GetKey(KeyCode.A))
                    {
                        vX = 1f;
                    }
                    else if (!Input.GetKey(KeyCode.D) && Input.GetKey(KeyCode.A))
                    {
                        vX = -1f;
                    }
                    else if (Input.GetKey(KeyCode.D) && Input.GetKey(KeyCode.A))
                    {
                        vX = 0f;
                    }
                    if (Input.GetKey(KeyCode.W) && !Input.GetKey(KeyCode.S))
                    {
                        vY = 1f;
                    }
                    else if (!Input.GetKey(KeyCode.W) && Input.GetKey(KeyCode.S))
                    {
                        vY = -1f;
                    }
                    else if (Input.GetKey(KeyCode.W) && Input.GetKey(KeyCode.S))
                    {
                        vY = 0f;
                    }
                }
                else
                {
                    vX = Input.GetAxis(horizontal);
                    vY = Input.GetAxis(vertical);
                }
                velocity.Set(vX, vY);

                float aX = 0f;
                float aY = 0f;
                if (keyboard)
                {
                    Vector2 mousePos = Vector2.zero;
                    if (Camera.main != null)
                        mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                    aX = mousePos.x - transform.position.x;
                    aY = mousePos.y - transform.position.y;
                }
                else
                {
                    aX = Input.GetAxis(r_horizontal);
                    aY = Input.GetAxis(r_vertical);
                }
                aim.Set(aX, aY);

                bool triggerPressed = false;
                if (keyboard)
                {
                    triggerPressed = Input.GetMouseButton(0);
                }
                else
                {
                    triggerPressed = Input.GetAxis(trigger) > triggerThreshold;
                }

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
