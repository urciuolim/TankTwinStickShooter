using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class BulletController : MonoBehaviour
{
    private float speed = 15;
    private float timeToLive = 10;
    [HideInInspector]
    public Vector3 velocity;
    private Rigidbody2D rb;
    [HideInInspector]
    public float damage = 1;
    [HideInInspector]
    public PlayerController shooter;
    public Collider2D shooterCollider;

    private void Awake()
    {
        JObject config = DriverController.instance.config;
        if (config["bullet_speed"] != null)
            speed = config["bullet_speed"].Value<float>();
        if (config["bullet_timeToLive"] != null)
            timeToLive = config["bullet_timeToLive"].Value<float>();
        if (config["bullet_damage"] != null)
            damage = config["bullet_damage"].Value<float>();

        /**
        if (DriverController.instance.verbose || firstTime)
        {
            firstTime = false;
            Debug.Log("Bullet speed set to " + speed);
            Debug.Log("Bullet ttl set to " + timeToLive);
            Debug.Log("Bullet damage set to " + damage);
        }**/
    }

    // Start is called before the first frame update
    private void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        velocity = transform.right * speed * DriverController.instance.fixedDeltaTime;//Time.fixedDeltaTime;
        shooterCollider = shooter.GetComponent<Collider2D>();
        //StartCoroutine(DestroyBullet());
    }

    // Update is called once per frame
    private void FixedUpdate()
    {
        rb.MovePosition(rb.position + new Vector2(velocity.x, velocity.y));
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        //if (!(collision.otherCollider.Equals(shooterCollider) || collision.collider.Equals(shooterCollider)))
        //{
            shooter.DestroyBullet(this.gameObject);
        //}
        //Destroy(gameObject);
    }

    IEnumerator DestroyBullet()
    {
        yield return new WaitForSeconds(timeToLive);
        shooter.DestroyBullet(this.gameObject);
        //Destroy(gameObject);
    }
}
