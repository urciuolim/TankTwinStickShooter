using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;
using System;

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
        if (DriverController.instance.verbose)
        {
            Debug.Log("Player " + playerID + " speed set to " + speed);
            Debug.Log("Player " + playerID + " trigger threshold set to " + triggerThreshold);
            Debug.Log("Player " + playerID + " reload time set to " + reloadTime);
            Debug.Log("Player " + playerID + " max health set to " + maxHealth);
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
    }

    // Update is called once per frame
    private void Update()
    {
        //GetInput();
        //Debug.Log("Update FPS: " + (1 / Time.deltaTime));
    }

    private void GetInput()
    {
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
                velocity.Set(Input.GetAxis(horizontal), Input.GetAxis(vertical));
                aim.Set(Input.GetAxis(r_horizontal), Input.GetAxis(r_vertical));

                if (Input.GetAxis(trigger) > triggerThreshold && canShoot)
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
        if (collision.gameObject.tag == "Bullet")
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
}
