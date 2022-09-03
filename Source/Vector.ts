

// creating custom vector class in hopes that it would be faster than vanilla JS array,
// although it uses js Array class under the hood
// In the future it could be modified to use Buffer class
class Vector{
    static default_size: number = 0;
    dim: number;
    c: number[];

    test(a: Vector){
        if(this.dim !== a.dim) throw "Dimension error";
    }

    constructor(n?: number | Vector, val?: number) {
        if(typeof n === 'undefined'){
            // default constructor
            this.dim = Vector.default_size;
            if(this.dim > 0) {
                this.c = new Array(this.dim);
            } else {
                this.c = null;
            }
        }
        else if(n instanceof Vector){
            // copy constructor
            this.dim = n.dim;
            if(this.dim > 0){
                this.c = new Array(this.dim);
            } else {
                this.c = null;
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
                    this.c = null;
                }
                Vector.default_size = n;
            }
            else{
                // constructor with initial value
                if(n == 0) n = Vector.default_size;
                if(n < 0) n = 0;
                this.dim = n;
                if(this.dim > 0){
                    this.c = new Array(this.dim)
                } else {
                    this.c = null;
                }
                Vector.default_size = n;
                for(let i = 0; i < this.dim; i++) this.c[i] = val;
            }
        }
    }

    size():number {return this.dim}

    resize(n: number){
        if(n != this.dim){
            this.dim = n;
            delete this.c;
            if(this.dim > 0){
                this.c = new Array(this.dim);
            } else {
                this.c = null;
            }
        }

    }

    put_value(val: number){
        for(let i = 0; i < this.dim; i++) this.c[i] = val;
    }

    at(i: number, val?: number): number {
        if (typeof val == "undefined") return this.c[i];
        else{
            this.c[i] = val;
            return this.c[i];
        }
    }

    eq(a: Vector | number): Vector{
        if(typeof a === "number"){
            for(let i = 0; i < this.dim; i++) this.c[i] = a;
        } else if(a instanceof Vector){
            this.resize(a.dim);
            for(let i = 0; i < this.dim; i++) this.c[i] = a.c[i];
        }
        return this;
    }

    copy(a: Vector){
        this.resize(a.dim);
        for(let i = 0; i < this.dim; i++) this.c[i] = a.c[i];
    }

    add_eq(a: Vector){
        this.test(a);
        for(let i = 0; i < this.dim; i++) {
            this.c[i] = this.c[i] + a.c[i];
        }
    }






}

