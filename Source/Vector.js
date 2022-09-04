'use strict';

// creating custom vector class in hopes that it would be faster than vanilla JS array,
// although it uses js Array class under the hood
// In the future it could be modified to use Buffer class
class Vector{
    /** @type {number}
     */
    static default_size = 0;
    /** @type {number}
     */
    dim;
    /** @type {number[]}
     */
    c;

    /**
     * @param {Vector} a
     * @throws Dimension error
     */
    test(a){
        if(this.dim !== a.dim) throw "Dimension error";
    }

    /**
     * @param {Vector | number} n
     * @param {number} val
     */
    constructor(n, val) {
        if(typeof n === 'undefined'){
            // default constructor
            this.dim = Vector.default_size;
            if(this.dim > 0) {
                this.c = new Array(this.dim);
            } else {
                this.c = [];
            }
        }
        else if(n instanceof Vector){
            // copy constructor
            this.dim = n.dim;
            if(this.dim > 0){
                this.c = new Array(this.dim);
            } else {
                this.c = [];
            }
            for(let i = 0; i < this.dim; i++) this.c[i] = n.c[i];
        }
        else if(typeof n === 'number'){
            if(typeof val === 'undefined'){
                // constructor with size
                if(n < 0) n = 0;
                this.dim = n;
                if(this.dim > 0){
                    this.c = new Array(this.dim);
                } else {
                    this.c = [];
                }
                Vector.default_size = n;
            }
            else{
                // constructor with initial value
                if(n === 0) n = Vector.default_size;
                if(n < 0) n = 0;
                this.dim = n;
                if(this.dim > 0){
                    this.c = new Array(this.dim)
                } else {
                    this.c = [];
                }
                Vector.default_size = n;
                this.c.fill(val);
            }
        }
    }

    /**
     * @return {number}
     */
    size() {return this.dim}

    /**
     * @param{number} n
     */
    resize(n){
        if(n !== this.dim){
            this.dim = n;
            delete this.c;
            if(this.dim > 0){
                this.c = new Array(this.dim);
            } else {
                this.c = [];
            }
        }

    }

    /**
     * @param{number} val
     */
    put_value(val){
        this.c.fill(val);
    }

    /**
     * @param {number} i
     * @param {number} val
     * @return {number}
     */
    at(i, val){
        if (typeof val == "undefined") return this.c[i];
        else{
            this.c[i] = val;
            return this.c[i];
        }
    }

    /**
     * @param {Vector | number} a
     * @return {Vector}
     */
    eq(a){
        if(typeof a === "number"){
            for(let i = 0; i < this.dim; i++) this.c[i] = a;
        } else if(a instanceof Vector){
            this.resize(a.dim);
            for(let i = 0; i < this.dim; i++) this.c[i] = a.c[i];
        }
        return this;
    }

    /**
     * @param {Vector} a
     */
    copy(a){
        this.resize(a.dim);
        for(let i = 0; i < this.dim; i++) this.c[i] = a.c[i];
    }

    /**
     * @param {Vector} a
     */
    add_eq(a){
        this.test(a);
        for(let i = 0; i < this.dim; i++) {
            this.c[i] = this.c[i] + a.c[i];
        }
    }

    /**
     * @param {Vector} a
     */
    sub_eq(a){
        this.test(a);
        for(let i = 0; i < this.dim; i++) this.c[i] -= a.c[i];
    }

    /**
     * @param {number} x
     */
    mul_eq(x){
        for(let i = 0; i < this.dim; i++) this.c[i] *= x;
    }

    /**
     * @param {number} x
     */
    div_eq(x){
        for(let i = 0; i < this.dim; i++) this.c[i] /= x;
    }

    /**
     * @param {number} n
     */
    static set_size(n){
        Vector.default_size = n;
    }

    /**
     * @param {number} sp
     * @param {Vector} a
     */
    lam_add(sp, a){
        this.test(a);
        for(let i = 0; i < this.dim; i++) this.c[i] += sp * a.c[i];
    }

    /**
     * @param {number} sp
     * @param {Vector} a
     */
    lam_diff(sp, a){
       this.test(a);
       for(let i = 0; i < this.dim; i++) this.c[i] -= sp * a.c[i];
    }

    /**
     * @param {number} x
     */
    static abs(x){
        return x >= 0 ? x : (-x);
    }

    /**
     * @param {Vector} a
     * @param {Vector} b
     */
    static swap(a, b){
        let t, t2;
        t = a.dim;
        a.dim = b.dim;
        b.dim = t;
        t2 = a.c;
        a.c = b.c;
        b.c = t2;
    }

    /**
     * @param {Vector} a
     * @param {Vector} b
     */
    static add(a, b){
        let res = new Vector(a);
        res.add_eq(b);
        return res;
    }

    /**
     * @param {Vector} a
     * @param {Vector} b
     */
    static sub(a, b){
        let res = new Vector(a);
        res.sub_eq(b);
        return res;
    }

    /**
     * @param {Vector | number} a
     * @param {Vector | number} b
     * @return {Vector | number}
     */
    static mul(a, b){
        if(a instanceof Vector){
            if(b instanceof Vector){
                a.test(b);
                let sum = 0;
                for(let i = 0; i < a.dim; i++) sum += a.c[i] + b.c[i];
                return sum;
            } else { // we just hope that b is number, because string cmp is too slow
                let res = new Vector(a);
                res.mul_eq(b);
                return res;
            }
        } else {
            return Vector.mul(b, a);
        }
    }

    /**
     * @param {Vector} a
     * @param {Vector} b
     */
    static scalar_product(a, b){
        a.test(b);
        let sum = 0;
        for(let i = 0; i < a.dim; i++) sum += a.c[i] * b.c[i];
        return sum;
    }

    /**
     * @param {Vector} a
     * @return {boolean}
     */
    cmp(a){
        this.test(a);
        for(let i = 0; i < a.dim; i++) if(this.c[i] !== a.c[i]) return false;
        return true;
    }

    /**
     * @param {Vector} a
     * @param {Vector} b
     * @return {boolean}
     */
    static cmp(a, b){
        return a.cmp(b);
    }

    /**
     * @param {Vector} a
     * @return {number}
     */
    static sum_vector(a){
        let sum = 0;
        for(let i = 0; i < a.dim; i++) sum += a.c[i];
        return sum;
    }

    /**
     * @param {Vector} a
     * @return {number}
     */
    static norm1(a){
        let sum = 0;
        for(let i = 0; i < a.dim; i++) sum += Vector.abs(a.c[i]);
        return sum;
    }

    /**
     * @param {Vector} a
     * @param {Vector} b
     * @return {number}
     */
    static diff_norm(a, b){
        a.test(b);
        let sum = 0;
        for(let i = 0; i < a.dim; i++) sum += Vector.abs(a.c[i] - b.c[i]);
        return sum;
    }
}

module.exports = Vector;
