export class Message {
    constructor() {
        this.el = document.getElementById('message');
        this.messageTimeout = null;
    }


    show(msg) {
        this.el.innerHTML = msg;

        if (this.messageTimeout) clearTimeout(this.messageTimeout);
        this.messageTimeout = setTimeout(() => {
            this.el.textContent = '';
        }, 3000);
    }
}